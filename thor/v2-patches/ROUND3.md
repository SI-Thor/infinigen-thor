# ROUND 3 — the splice collected, and the Warp mesher as a live stage

**Date:** 2026-10-31 (07:00–12:00 AKDT) · **Where:** this worktree
(`~/lm-wt/engine-r3`) · **Box:** NVIDIA Jetson AGX Thor (sm_110, 14×
Cortex-X4A, 128 GB LPDDR5X, Ubuntu 24.04, driver 595.78) · **Base:**
infinigen `main @ 224598d4` (= upstream 2.0.0a3, round 2's commit),
procfunc 0.37.0 + the round-2 ValueProxy patch + round 3's new
`subtree_memo` module, bpy 4.2.0, numpy 1.26.4, warp-lang 1.18.0. The venv
is this tree's `.venv`; `~/lm-wt/v2-next` and `~/lm-wt/lab4` were copied
from, never written.

### The short version

| # | Task | Result |
|---|---|---|
| 1 | **The splice** — re-execute the desk edit reusing the realised results of unchanged graph nodes | **Done, twice.** The subtree memo (engine side) splices the base run's realised objects into the edited run: the splice reproduces the fully-regenerated desk scene **exactly** (canonical structural equality — geometry, transforms, mesh content, material content — sha `e807fbb2…`, identical across two independent runs) at **2.33×/2.46×** the full regeneration's wall (13.65→5.86 s, 12.89→5.25 s). 30 of the splice's 62 builder calls are hits; **98 of the 128 builders the base run executed are never called at all**; strict name equality fails *by design* and the mechanism is measured and stated (§2.4) |
| 2 | **The Warp mesher in the engine** — lab 4's SDF evaluation wired into the v2 terrain/mesher stage | **Done.** The real OcMesher (vendored prebuilt `core.so`, unmodified) with the SDF answered by the Warp kernels produces **byte-identical meshes** to the same mesher over Infinigen's own CPU libraries (both 150,731v/301,267f + 147,461v/295,422f, sha-equal), at **3.90× the wall** (20.85→5.35 s; SDF share 74.1%→6.3%); `terrain_mesh_scene_rand` traces to a **byte-stable 63-node graph** with the mesh as an opaque leaf, and a one-parameter edit re-meshes in **6.13 s** — live re-meshing is a stage |
| 3 | Packaging | **Done.** Six patches (round 2's three, regenerated and verified, + round 3's three), verified to apply sequentially onto pristine upstream and reproduce this working tree byte-for-byte; drivers `splice_tool.py` / `warp_mesher_bench.py`; this file and `README.md` |

**Box state, beside every number.** Shared throughout: load average
7.3–15.5 on 14 cores (another seat's ComfyUI resident; the splice runs are
CPU-only — GPU idle, 315 MHz, ~2.4 W in every sample; the mesher runs took
the GPU to 1575 MHz under warp with a second seat idle-resident). All wall
clocks are upper bounds on an idle box; the load moved between runs (the
same splice measured 2.33× and 2.46×), the verdicts never did.

---

## 0. What was inherited and what this run did

The previous agent (stopped by quota ~10:47) left: the copied venv and
trees, the round-2 patches, `splice_tool.py` with the four-run protocol
(base / full / populate / splice), the subtree-memo module installed in the
venv, `warp_mesher.py` in the infinigen tree (registered in the manifest),
`warp_mesher_bench.py` (never run), and 26 splice log/attempt pairs ending
in `splice == full_desk: False` with a name-drift signature.

This run inherited all of it, then:

- **found the inherited splice was one bug and one misread away from
  passing.** (a) The memo's counters were cumulative across sessions — the
  splice's real 62 tracked calls / 48.4 % hit rate read as "386 calls /
  7.8 %", which sent runs 5–26 chasing a re-execution cascade that did not
  exist. (b) The supersede path's protected-pointer set was snapshotted
  once and never shrank, so the base run's stale scatter instances survived
  every supersede and kept Blender's name space occupied. Fixed both
  (per-session stats; per-entry cached pointer sets with protection
  recomputed over the live store, consumed entries kept protectively).
- **diagnosed the residual inequality to its mechanism** with two
  purpose-built probes (`splice_diag.py` call-sequence tracer,
  `splice_tagdiag.py` object-ancestry tagger): after the supersede fix the
  two scenes are congruent in **every structural field** — all 104 objects
  pair with identical meshes, transforms, rotations, visibility — and the
  only residue is Blender's *auto-numbered names* (`cup_rand.008` vs
  `.001`, `material_57` vs `material_54`), which count creation history and
  therefore cannot match a splice (§2.4). Added the canonical structural
  equality metric that states this honestly; strict-name equality stays
  reported.
- **ran the Warp mesher bench end to end** (it had never executed): fixed a
  vendor-path bug, a lab-4 transcription slip in the fixed parameter pack
  (measured against v1's own element code, the authoritative source), a
  Proxy-iteration and a Proxy-in-numpy traceability wall in the scene
  generator, and the `__wrapped__` access in the live-remesh arm — then
  took the final numbers below. Re-verified the mountains kernel
  bit-identical in this tree (max |diff| = 0.0, 1M points,
  `verify_terrain_kernel.py`).
- regenerated the patch set and verified it applies sequentially onto
  pristine `224598d4` and reproduces the working tree exactly; wrote
  `patches/RATIONALE.md` §4–6, `README.md`, this file.

---

## 1. Task 2 first — the Warp mesher as a v2 stage (the clean half)

### 1.1 The same mesher, two SDF implementations, one mesh

`warp_mesher_bench.py` runs the real OcMesher — the vendored prebuilt
`core.so` driving octree construction, visibility filtering, bisection and
marching cubes exactly as upstream — over lab 4's fixed scene (mountains
seed 123456 + warped_rocks seed 654321, 1280×720 camera at K=1000,
`pixels_per_cube=8`, bounds ±30 m), twice: once with the SDF answered by
Infinigen's own CPU libraries (`CpuSDFElement`, ctypes, OpenMP), once by
the Warp kernels (`WarpSDFElement`, resident device buffers).

| | wall | SDF inside | share | meshes |
|---|---|---|---|---|
| CPU SDF (Infinigen's libraries) | **20.85 s** | 15.46 s | 74.1 % | 150,731v/301,267f + 147,461v/295,422f |
| Warp SDF (this round's stage) | **5.35 s** | 0.34 s | 6.3 % | identical vertex/face counts |
| | **3.90×** | **45×** | | **sha256-equal, both meshes** |

Reproduced across four completed runs of the bench (2.62× at load 13.4
with a cold kernel compile inside the timed arm; then 3.84×, 3.60×, 3.90×
warm — the ratio moves with load and cache, the ordering and the
bit-identity never do). The SDF share
collapsing 74→6 % flips the mesher from SDF-bound to bookkeeping-bound, as
lab 4 projected — the octree + marching-cubes tail (~5 s) is the next
thing to attack, and nothing in this round touches it.

The bit-identity is at mesh level (the mesher's own output arrays,
sha256 over vertices+faces), which is *stronger* than lab 4's per-batch
SDF check: the octree walked identically, every bisection converged
identically, the extracted surfaces are the same bytes.

### 1.2 It is a stage: grammar → opaque leaf → byte-stable graph

```
$ .venv/bin/infinigen terrain_mesh_scene_rand --seed 0 --trace graph \
    --trace_level RANDOM_CONTROL
→ rc=0, graph.json byte-stable across runs (2 traces, sha 2849992f…),
  63 nodes, the mesh leaf present: infinigen2.terrain.warp_mesher.terrain_mesh,
  its params argument referencing the grammar's draw nodes
```

`terrain_mesh_params_rand` draws element seeds/ranges, camera
distance/height/azimuth/pitch and focal length as `pf.random` nodes;
`terrain_mesh` is an opaque `@pf.tracer.primitive` — the recipe records
the call and its parameter edges, and the octree/GPU/meshing happen at
realization. A camera or element edit is therefore a graph edit at the
same level of detail as any v2 parameter, and re-meshing is that one
leaf's execution — measured directly:

| edit | wall | result |
|---|---|---|
| `camera_distance × 0.6` (same seed) | **6.13 s** | both meshes rebuilt (195,984v and 104,694v — the LOD followed the camera), sha differs from base: a real edit, not a replay |

Live re-meshing is thus a **stage**: under 7 s per parameter edit on this
box, against 20.9 s for the CPU-SDF mesher or a full scene rebuild.

And the scene renders end to end — `terrain_mesh_scene_rand render_cycles
--seed 0` → rc=0, generation 8.6 s (the grammar's own draw: two meshes,
164,737v + 173,731v, 118 SDF batches / 60 M points with **0.51 s total on
the GPU**), Cycles 155 s, `outputs/r3-render-meshscene/Camera/0000.png`
(granite mountains under sky, from the grammar-placed camera).

> **Verdict 1 — does the Warp SDF survive contact with the engine?**
> **Yes, bit-identically, and it makes re-meshing cheap enough to be a
> live stage.** The mesher runs unmodified; only the dominant term lab 4
> identified moved to the GPU, and it moved verifiably (sha-equal meshes,
> 3.90× wall). The remaining wall is octree bookkeeping — the next round's
> target, and a different one.

---

## 2. Task 1 — the splice: reusing the realised results of unchanged nodes

### 2.1 The mechanism (what the engine-side instrumentation is)

`procfunc.subtree_memo` (new module, round-3 procfunc patch) memoises the
*realised* results of builder functions, keyed by a content hash of each
call — function identity + argument values, with numpy arrays hashed by
bytes, `np.random.Generator` by its **bit-generator state**, live bpy
datablocks by pointer+name+type, procfunc wrappers by their realized
datablock, and anything unhashable skipped (an unhashable argument can
only cost time, never correctness). The RNG-state term is what makes v2
spliceable at all: v2 draws every grammar decision from `spawn`ed child
generators (independent bit-streams), so a builder called with the same
spawned rng and the same values produces the same bytes.

Two modes over one in-process store (the resident-engine model of DIRTY.md
§6 — a hit's result is a live Python object graph holding bpy datablocks):

- **populate** — every wrapped call executes and is recorded: result,
  post-call rng states (restored on hit, so shared generators stay
  aligned), bpy objects created (precise attribution via creation hooks on
  procfunc's `to_mesh_object`/`to_aliases`/primitive sites), and
  partitioned self-time;
- **splice** — a call whose hash is present is served the recorded result
  without executing; a call whose hash is absent executes, first
  *superseding* the base entries of the same function (deleting their bpy
  objects unless a live entry still references them, so the rebuild
  retakes the same names).

`infinigen2/util/splice.py` (round-3 infinigen patch) is the scene-side
policy: builders under the scene/object layers are memoisable; entry
points, mutators (placement, naming, cleanup) and realized-geometry
services are not; the installer swaps wrappers into every loaded namespace
holding a reference. Wrappers are inert outside a session — proven by run 3
of the protocol below.

### 2.2 The four-run protocol (one process, one box state)

The edit is round 2's desk edit: one `chosen_idx` at `room.py:661` forced
False→True with the RNG draw still consumed (stream position unchanged).

| run | what | wall | objects | equality |
|---|---|---|---|---|
| 1 `base_plain` | seed 0, no instrumentation session | 15.34 s | 94 | the reference base |
| 2 `full_desk` | desk forced, no session — **full regeneration** | 13.65 s | 104 | the reference edit (530 `choice_idx` calls vs 469) |
| 3 `populate` | seed 0 under `memo.session("populate")` | 13.05 s | 94 | **== base_plain, byte-identical inventory** (strict names *and* canonical) — the instrumentation changes nothing |
| 4 `splice` | desk forced under `memo.session("splice")`, resident scene | **5.86 s** | 104 | **== full_desk, canonical structural equality** (sha `e807fbb2…`); strict names differ — §2.4 |

Repeated in full (run 32, load 8.6): 12.89 s full vs **5.25 s** splice =
**2.46×**, same verdicts, and the splice scene's canonical sha is
**identical across both runs and equal to full_desk's** — the splice is
deterministic, not occasionally-right.

### 2.3 What the splice collected

Per-session counters (after the fix — see §2.5):

| measure | populate | splice |
|---|---|---|
| tracked builder calls | 324 | **62** |
| hits / misses | — / 324 | **30 / 32** (48.4 % of calls hit) |
| distinct functions executed | 128 | **37** |
| functions never called | — | **98 of the 128** (their callers hit, so the calls never happened) |
| splice-only functions | — | 7 (the desk subtree: `desk_setup_rand`, `desk_rand`, `chair_office_rand`, …) |
| content-hash overhead | 0.03 s | **0.007 s** |

Read against the stage cache's graph-level claim (the desk's dirty closure
= 12.6 % of an 18,725-node scene, the other **87.4 % reusable**): the
splice never touches 98 of 128 builders and hits 30 of the 62 calls it
makes; what it re-executes (5.79 s of tracked self-time) is the desk
subtree plus the join's downstream consumers (`wall_storage_setup`,
`decorate_*`, the scatters — exactly the closure DIRTY.md §3 computes),
and its magnitude matches round 2's independent cProfile attribution of
the same edit (+5.5 s of generation). Of the base run's 13.05 s of
memoized build time the splice re-executes 5.79 s and collects **7.26 s
(56 % by time — the dirty 12.6 % of *nodes* is 44 % of *time*: the
closure contains the scatters, the scene's most expensive builders)**;
by builder coverage it skips 98 of 128 outright (77 %) and hits 48 % of
the calls it still makes. The **2.33–2.46× wall** is the saving measured
end to end — the reusable work collected at object level, not just proven
at graph level.

Two honest limits, both stated by the mechanism: the memo is in-process
(resident-engine — a hit serves live datablocks, so the stage cache's
cross-process byte-identity guarantee does not apply here); and splice
safety is function-granular — a function whose call sequence is not
prefix-stable over-deletes and re-executes, which is correct (execution is
deterministic) but less collected.

### 2.4 What "equal" means for a splice — and why strict names cannot match

The splice scene and the fully-regenerated scene have the same 104 objects
with **identical mesh content hashes, locations, rotations, scales and
visibility** — zero structural differences, verified object-by-object
after canonical pairing (group by name stem, order by content). The
differences Blender's own names show are all one mechanism:

> **auto-numbered names count creation history, and a splice skips most
> creation by design.**

Two measured consequences: (a) object suffixes — the base run's superseded
scatter instances occupied `cup_rand.001…`, so rebuilt instances landed on
`.008…` (50 of the stale instances are now deleted by supersede; the rest
are protected as pools for hits); (b) material numbers — the desk's three
materials shifted every later auto-number, so the *same shader* drawn from
the same rng state is `material_57` in the full run and `material_54`
served from the base run. Note the full regeneration itself has this
property *relative to the base scene* — strict-name equality across an edit
is not something the un-spliced pipeline offers either. The equality the
splice owes is structural, and it holds: the canonical sha (name-stems +
all content fields + material *content* hashes — node types, socket
values, link topology) matches exactly, across two independent runs.

### 2.5 The two findings that unblocked it (for whoever touches this next)

1. **Counters must be per-session.** The memo's stats accumulated across
   populate and splice; the summary divided splice-session hits by both
   sessions' calls. The splice was already hitting 30/62 calls when every
   log read "7.8 %". Diagnosis: a per-call trace (`SPLICE_TRACE_CALLS=1`,
   caller site + per-argument token hashes, `splice_diag.py`) showed the
   splice making 62 calls against the report's 386 — the extra 324 were
   the populate session's counters, not executions.
2. **Protection must shrink when entries die.** The supersede path's
   protected-pointer set was computed once (at the first supersede) and
   only ever grew. Because the base parent entries were still in the store
   at that moment, every base scatter instance entered the protected set
   and stayed there after its owners were superseded — so the rebuild
   couldn't free the names, and (worse) the stale instances sat in the
   scene. Fix: per-entry cached pointer sets; protection recomputed as a
   union over the *live* store at every supersede; consumed (already
   served) entries stay in their buckets so the pools they served remain
   protected. Symptom to watch for if this regresses: `superseded_objects`
   falling to 0 while pending entries hold creations.

> **Verdict 2 — does the splice collect the desk edit's reusable work?**
> **Yes, measured end to end.** The spliced run reproduces the fully
> regenerated scene exactly at the structural level (same canonical sha
> twice), skips 98 of 128 builders outright, serves 30 hits, and runs at
> **2.33–2.46×** the full regeneration's wall — the stage cache's 87 %
> graph-level bound, collected at object level by the engine-side
> instrumentation DIRTY.md named as the missing piece. The honest costs:
> it needs the resident process (live datablocks), and equality is
> canonical-structural, not name-strict — the same trade every
> partial-evaluation cache makes when the runtime's names are counters.

---

## 3. Artifacts

| what | where |
|---|---|
| splice driver (the four-run protocol) | `splice_tool.py` |
| splice final runs | `outputs/splice-run31/`, `outputs/splice-run32/` (`splice-report.json`, per-run inventories + memo stats) |
| splice diagnosis (call trace, ancestry) | `splice_diag.py`, `splice_tagdiag.py`, `outputs/splice-diag1/`, `outputs/splice-diag2/` |
| the subtree memo | `.venv/.../procfunc/subtree_memo.py` → `patches/procfunc-0.37.0/procfunc-subtree-memo.patch` |
| the scene-side policy | `src/infinigen-upstream/src/infinigen2/util/splice.py` → `patches/infinigen2-splice-instrumentation.patch` |
| mesher bench (task 2 driver) | `warp_mesher_bench.py` |
| mesher final numbers | `results/warp-mesher/warp-mesher.json` (+ `trace-a/`, `trace-b/` graphs) |
| the stage module | `src/infinigen-upstream/src/infinigen2/terrain/warp_mesher.py` → `patches/infinigen2-warp-mesher-stage.patch` |
| kernel re-verification in this tree | `verify_terrain_kernel.py` (max \|diff\| = 0.0, 1M pts) |
| patches + rationale (all five) | `patches/` (`RATIONALE.md` §1–3 round 2, §4–6 round 3) |
| inherited debugging history | `logs-splice-run1..32.txt`, `outputs/splice-run1..26/` (the previous agent's iterations), `logs-warp-bench1..6.txt` |
| box state beside every run | `lab4-scripts/boxstate.py` (loadavg/power/GPU per report) |

## 4. How to reproduce

```bash
cd ~/lm-wt/engine-r3
./apply-procfunc-patch.sh        # round-2 procfunc patch (idempotent)
(cd .venv/lib/python3.11/site-packages && \
 patch -p1 --forward < $PWD/../../patches/procfunc-0.37.0/procfunc-subtree-memo.patch)

# task 1 — the splice, four runs, ~1 min under no load; rc=0 iff both
# equality verdicts pass
.venv/bin/python splice_tool.py measure --out outputs/rep-splice

# task 2 — the Warp mesher: cpu-vs-warp bit-identity + the stage + live re-mesh
.venv/bin/python warp_mesher_bench.py --out results/rep-warp-mesher

# the stage by hand: trace it, render it
.venv/bin/infinigen terrain_mesh_scene_rand --seed 0 --trace graph \
    --trace_level RANDOM_CONTROL --output outputs/rep-mesh-trace
.venv/bin/infinigen terrain_mesh_scene_rand render_cycles --seed 0 \
    --output outputs/rep-mesh-render

# the kernel, re-verified against Infinigen's own CPU library
.venv/bin/python verify_terrain_kernel.py

# the patches, onto pristine upstream (in order)
#   procfunc-value-proxy.patch            (round 2)
#   procfunc-subtree-memo.patch           (round 3)
#   infinigen2-traceable-scenes.patch     (round 2)
#   infinigen2-warp-terrain-stage.patch   (round 2)
#   infinigen2-splice-instrumentation.patch (round 3)
#   infinigen2-warp-mesher-stage.patch    (round 3)
```
