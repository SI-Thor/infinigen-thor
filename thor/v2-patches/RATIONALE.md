# Why each patch exists

Six patches make rounds 2 + 3 work. They are ordered by layer: the tracer
capability first (procfunc), the scene-code traceability second (infinigen2),
the terrain stage third (infinigen2) — round 2's three — then round 3's three
engine-side additions: the subtree memo (procfunc) with its scene-side
target policy (infinigen2), and the Warp-meshed terrain stage (infinigen2).
Each is independently upstreamable and each says what it does *not* try to
do.

## 1. `procfunc-0.37.0/procfunc-value-proxy.patch` — the tracer capability

**Problem.** procfunc 0.37.0's tracer refuses to look at any value it cannot
statically know. Two consequences killed v2's *scene* generators:

1. *RNG stream divergence.* At `RANDOM_CONTROL` a `pf.random.*` draw was
   recorded as a node and never executed, but `pf.control.choice` always
   consumes the real generator. After the first skipped draw, the choices the
   tracer records are not the choices the renderer takes (probes
   `probe_rng_stream_micro.py`, `probe_choice_stream.py`).
2. *No values at all.* Scene generators are imperative: they branch on
   sampled counts (`rng.spawn(n)`), index lists with sampled indices, unpack
   results, and iterate what `choice` returns. With every one of those values
   an opaque `Proxy` (`Proxy does not support __iter__ / __bool__`), the trace
   dies at the first data-dependent statement.

**Fix — one concept, `ValueProxy`: a Proxy that also carries the concrete
value it stands for.** The tracer already *has* many of these values; it just
threw them away:

- A leaf call whose only dynamic input is the RNG (`pf.random.uniform(rng, a,
  b)` with concrete `a, b`) is evaluated for real and returns a `ValueProxy`.
  The draw is recorded as a node *and* consumed from the generator, so the
  tracer's stream stays identical to the renderer's, and the traced program
  can use the sampled value (`probe-rng-stream-micro.json`: traced
  `chosen_idx` = real draw at `RANDOM_CONTROL`).
- The same for leaf calls whose proxy args are all `ValueProxy`s, and for
  generator (`RngProxy`) method calls such as `rng.choice(len(names), size=n)`
  — one dynamic input, the generator, so evaluate it.
- At `RANDOM_CONTROL`, `choice` peek-through already executes *every* option;
  the chosen branch's result is therefore known and is returned as a
  `ValueProxy` (non-callable options return their value directly, so
  `if pf.control.choice(...)` works). Iterating a chosen branch's list,
  unpacking its NamedTuple, reading its fields — all work again.
- Arithmetic, indexing and attribute access propagate known values
  (`_known_getattr` / `_maybe_value_proxy`), with one invariant: a `ValueProxy`
  never wraps a `Proxy`. A Proxy result already has its own (more precise)
  node; hiding it would break calls on it.

**Deliberately not done:** no attempt to evaluate anything that reads realized
Blender geometry — those values genuinely do not exist while tracing, and
patch 2 handles them at the scene layer instead.

## 2. `infinigen2-traceable-scenes.patch` — scene code the tracer can capture

**Problem.** v2's scene layer interleaves two kinds of code: *grammar*
(choices, parameter draws, composition — exactly what the recipe should
record) and *runtime geometry services* (collision tests, bbox reads,
snapping, culling, realized-datablock identity — code whose result depends on
meshes that only exist at render time). The tracer cannot express the second
kind, so `room_livingroom_rand --trace graph` died at
`room_shape.py:133` (`for idx in np.unique(read_attribute(...))`), and after
that at a dozen more sites of the same class.

**Fix — one rule, applied per kind.** The rule lives in one new helper,
`infinigen2/util/tracing.py::is_tracing()` (procfunc already exposes the
tracing flag as `pf.context.globals.current_trace_level`; `pf.control.choice`
itself branches on it).

- *Statically-knowable control flow is rewritten to be static.* The room
  outline is always a rectangle, so it always has exactly 4 walls:
  `ROOM_WALL_COUNT` in `curves/room_edge.py` replaces the loop over
  `np.unique(<runtime attribute read>)` in `room_shape.py::_split_flat_walls`.
  (Verified for seeds 0–39 by `probe_wall_groups.py`: the unique wall indices
  are always {0,1,2,3}.)
- *Runtime geometry services declare trace-time assumptions.* Predicates
  return the optimistic answer (`intersection_test → False`, `clear_of →
  True`, grounded-checks → `True`); filters assume everything placed is kept
  (`keep_non_colliding → (non-None items, colliders)`); retries take the
  first attempt (`retry_place` places once and returns the child); realized
  identity partitions have no answer and yield no targets (surface/small
  decoration defers to render time). Every assumption is a comment at the
  guard and a `docs/` note in ROUND2.md. Because v2 draws every grammar
  decision from `rng.spawn(...)` children (independent numpy bit-streams), a
  divergence inside one placement child cannot cascade into sibling choices.
- *Pure realized side effects become opaque leaves* (`@pf.tracer.primitive`):
  `snap_to_plane`, `collision_set`, `propagate_modifiers_to_instances`,
  `delete_object(s)`. The recipe records the call and its argument nodes; the
  geometry work happens at render time, exactly like modifier baking.
- *Cosmetic realized writes are skipped* (`name_objects`, datablock renames).
- Two one-line mechanical rewrites remove Proxy iteration where the same
  semantics is expressible with operators (`[*a, x] → a + [x]`, a naming loop
  → `name_objects`).

**Render-equivalence proof.** Every change is guarded by `is_tracing()` or is
a decorator (inert outside a trace), so the un-traced pipeline must be
unchanged. Verified two ways: object-level, a 94-object scene inventory
(`/tmp/scene_inventory.py` pattern) is byte-identical to the same seed in an
unpatched tree at the same commit; pixel-level, the seed-0 render differs
from round 1's unpatched baseline by **1 pixel of ±1 LSB** (GPU accumulation
noise). An earlier draft of this patch moved
`overlap_wall_plane_edges` to a different point in the room build and shifted
skirting geometry — the inventory diff caught it (one mesh, 84 verts), which
is exactly the check to keep.

## 3. `infinigen2-warp-terrain-stage.patch` — Warp `mountains` as a v2 stage

**Problem.** v2 has no terrain. The lab-7 Warp port of v1's `mountains`
element is bit-identical to Infinigen's own CPU/CUDA libraries and 6× its CUDA
kernel, but it is a benchmark script, not a stage that participates in the
recipe graph.

**Fix.** `infinigen2/terrain/warp_mountains.py` wraps the *verbatim* kernel
(the module's kernel is re-verified bit-identical against the vendored
`mountains_cpu.so` over 1M points, `max|diff| = 0.0`) in the same two layers
the scene patch uses:

- `mountains_params_rand` — the grammar: draws the element's parameters with
  `pf.random` using v1 landtiles' ranges (`min_freq` U(5e-4, 2e-3), `height`
  U(60, 90), …), so every terrain parameter is a seed-controlled node.
- `mountains_heightfield` — an opaque `@pf.tracer.primitive`: evaluates the
  element on the GPU (`wp.launch` over a 512×512 grid, ~262k vertices) and
  builds the displaced plane. In the traced graph the terrain is one node
  whose args reference the six parameter-draw nodes.
- `terrain_scene_rand` — a Scene generator composing the stage with
  Hosek-Wilkie sky and a grammar-driven camera; registered in `manifest.json`
  alongside the Object-category `mountains_terrain_rand`.

The Warp kernel stays byte-for-byte the lab-7 port (including its hard-won
quirks: `FastFloor` truncation, float-precision `log_uniform`, the `b`-not-
`c` rotate in `hash_bit_final`); only the wrapper is new.

## 4. `procfunc-0.37.0/procfunc-subtree-memo.patch` — the engine-side splice (round 3)

**Problem.** The stage cache (docs/research/stage-cache-thor.md) proved the
desk edit's reusable work at graph level — the dirty closure is 12.6 % of an
18,725-node scene — but the venv boundary re-traces a stage as a unit, so
the other 87 % is re-executed every edit. Collecting it needs an engine-side
memo over the *realised* results of the functions a scene is composed from:
DIRTY.md §3's named missing piece.

**Fix — one module, `procfunc.subtree_memo`, plus the scene-side policy in
patch 5.** The memo keys a *call* by a content hash of (function identity,
argument values), where numpy arrays hash by bytes, `np.random.Generator`
by its bit-generator state (the property that makes v2 spliceable: every
grammar decision draws from a `spawn`ed child generator, so same spawned
rng + same values ⇒ same bytes), live bpy datablocks by pointer+name+type,
and procfunc object wrappers by their realized datablock. Anything
unhashable skips the memo and executes — an unhashable argument can only
cost time, never correctness. Two modes over one in-process store (the
resident-engine model of DIRTY.md §6): **populate** executes and records
(result, post-call rng states, bpy objects created, self time), **splice**
serves a recorded result and restores the recorded rng post-states, or
executes and *supersedes* the base entries of the same function — deleting
their bpy objects first so the rebuild retakes the same names.

Two hard-won rules live in the code comments and are worth restating:

- **Protection must shrink.** An object may be deleted by supersede iff no
  *live* store entry references it. The first implementation snapshotted the
  protected set once (at the first supersede) and never shrank it, so the
  base run's stale scatter instances stayed alive through the whole splice —
  occupying Blender's name space, pushing every rebuilt object onto a
  shifted numeric suffix. The fix: per-entry cached pointer sets, protection
  recomputed as a union over the *current* store at every supersede, and
  consumed (already-served) entries kept in their buckets so the pools they
  served stay protected.
- **Counters are per-session.** The memo's stats reset at each session: a
  process that populates and then splices must not report the two sessions'
  calls as one pile. Reading the cumulative counters made a 48 %-of-calls
  splice look like 7.8 % for a day of debugging (ROUND3.md §2).

**Deliberately not done:** no cross-process persistence (a hit's result is a
live Python object graph holding bpy datablocks — this memo is the resident
engine, not a store); no eviction (the session's lifetime is the budget);
no attempt to memoise mutators (placement, naming, cleanup — the scene-side
policy in patch 5 is the guard).

## 5. `infinigen2-splice-instrumentation.patch` — which builders the memo may wrap

**Problem.** The memo is only sound for functions whose whole effect is the
return value (plus rng consumption, which the memo records). Infinigen's
scene layer is full of functions that mutate the scene, read realized
geometry, or rename datablocks; wrapping one of those would serve a stale
side effect.

**Fix — one new module, `infinigen2/util/splice.py`, the policy half:**
every *builder* under the scene and object layers is memoisable; entry
points, mutators and realized-geometry services are not. The installer wraps
each target in every loaded namespace that holds a reference (the defining
module and every importer — the same reference-patching problem procfunc's
tracer solves with search scopes). Wrappers are inert unless a memo session
is active, so an un-instrumented generation is byte-for-byte what it was
(proven: the populate run's scene inventory is byte-identical to an
un-instrumented run's — ROUND3.md §2). The target list is overridable with
`INFINIGEN_SPLICE_TARGETS` (JSON list of `module:qualname`, or `module:*`).

**Deliberately not done:** no static analysis of which functions are pure —
the curated module list is the measured set, with the policy stated as
comments; no wrapping of the placement services (they read realized
geometry mid-placement, exactly the class round 2 taught the tracer to
treat as opaque).

## 6. `infinigen2-warp-mesher-stage.patch` — the OcMesher as a v2 stage (round 3)

**Problem.** Lab 4 (docs/lab/exp-warp-mesher-sdf.md) proved the OcMesher is
SDF-evaluation-bound (~69 % of wall) and that the two terrain elements are
bit-identical in Warp at 75–85× the CPU libraries — but as a benchmark
script, not an engine stage: v2 had no way to mesh a terrain, let alone
re-mesh one after a parameter edit.

**Fix — `infinigen2/terrain/warp_mesher.py`, three layers, the same shape
as round 2's terrain stage:**

- the numeric kernels stay verbatim (`mountains_kernel` reused from
  `warp_mountains`; `warped_rocks_kernel` is lab 4's port, including its
  one-argument-hash `log_uniform` trap);
- `WarpSDFElement`/`CpuSDFElement` adapt an element to the OcMesher's
  kernel ABI — the Warp side with resident device buffers (the difference
  from Infinigen's own CUDA library, which `cudaMalloc`s per call), the CPU
  side over Infinigen's own `.so` libraries as the verification oracle and
  timing baseline;
- `terrain_mesh_params_rand` (grammar: element seeds/ranges, camera draws,
  LOD) and `terrain_mesh` (an opaque `@pf.tracer.primitive` that runs the
  real vendored OcMesher with the SDF on the GPU and links the meshes into
  the scene) — so a camera or parameter edit is a graph edit at the same
  level of detail as any v2 parameter, and re-meshing is one leaf's
  execution. `terrain_mesh_scene_rand` composes it with sky and material
  into a renderable scene, registered in `manifest.json`.

The mesher itself is *not* ported: the vendored prebuilt `core.so` runs
as-is (nothing in `~/src/infinigen-thor` is written; the vendored copies
live under `vendor/`). What moves to the GPU is the dominant term lab 4
identified — the SDF evaluation — which is exactly the part that is
bit-identical.
