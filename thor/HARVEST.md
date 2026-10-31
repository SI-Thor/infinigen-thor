# HARVEST: the cheap fork harvest (plan item 6)

Date: 2026-10-31 · Branch: `thor-harvest` (from `thor-spike` @ `5d1c4455`) ·
Box: Jetson AGX Thor Developer Kit, aarch64, gcc 13.3, CUDA 13.2 at
`/usr/local/cuda`, Blackwell sm_110 · Source of truth:
`~/src/builder-ide/docs/research/forks.md` (the fork scout's nugget list) and
`~/src/builder-ide/docs/plan.md` draft 2, week 1 item 6.

**Not pushed. No GitHub Actions touched. No sub-agents. The main tree's
`.venv` and the running overnight seed sweep were not touched** — every
build and test below ran in a scratch venv (`.venv-harvest/`) and in
scratch directories under `/tmp`.

## Verdict

Plan item 6 asked for three things: the two `#include` fixes upstream has not
merged, the Boolean EXACT-solver commit that can otherwise hang a night, and
the `sdf_from_mesh` empty-mesh GPU fix. **All of them are in**, cherry-picked
with `-x` (authorship and attribution kept), plus the `NULL` → `nullptr`
compile fix that lives in the same header as the first one. Each one has a
regression test that **fails without the fix and passes with it**, the whole
terrain C++/CUDA source set still compiles on the Thor, and the licence is
unchanged BSD-3-Clause.

| # | What | Fork | Branch | Upstream commit | Upstream PR | Our commit |
|---|---|---|---|---|---|---|
| 1 | `FastNoiseLite.h`: add `<cstddef>` + `<cassert>` | [ebajemon-tw3](https://github.com/ebajemon-tw3/infinigen) | `macos-arm64-fixes` | [`0eb315a0`](https://github.com/ebajemon-tw3/infinigen/commit/0eb315a047a711be81e28e077cb167bf4d584b12) | not filed | `a5fd73b8` |
| 2 | `FastNoiseLite.h`: `NULL` → `nullptr` (device code) | [DiegoGomesDG](https://github.com/DiegoGomesDG/infinigen) | `fix/null-compilation-error-v1` | [`b6ee485`](https://github.com/DiegoGomesDG/infinigen/commit/b6ee4853a1b63a0982585c14b47a0a97f6cb9dae) | [#508](https://github.com/princeton-vl/infinigen/pull/508) | `fa987508` |
| 3 | `populate.apply_cutter`: Boolean FAST → **EXACT** | [folouiseAWS](https://github.com/folouiseAWS/infinigen) | `fix/populate-exact-boolean-solver` | [`d52fac4`](https://github.com/folouiseAWS/infinigen/commit/d52fac4963afb2bd9b718cfb2a04a2aaa6aee16d) | [#498](https://github.com/princeton-vl/infinigen/pull/498) | `dc20985e` |
| 4 | `sdf_from_mesh`: empty mesh no longer aborts | [DiegoGomesDG](https://github.com/DiegoGomesDG/infinigen) | `fix/cuda-terrain-sdf` | [`3aa7b1a`](https://github.com/DiegoGomesDG/infinigen/commit/3aa7b1a6fbab8782aa394baf00d2b1927d362206) | [#504](https://github.com/princeton-vl/infinigen/pull/504) | `1a4040c9` |
| 5 | the same include fix in the **second** copy of the header | (same as #1) | — | (hand-applied) | — | `080eb671` |

Plus our own: `fc393bfd` (the regression tests) and `134a00df` (gitignore the
scratch venv).

**Why these four and not the rest of the list.** The scout's list is 730 forks
deep; the plan's item 6 names the cheap, near-zero-risk end of it. Items #1–3
of the scout's "do now" order are exactly #1, #2 and #3 above; #4 is the
third item of that order and the plan names it explicitly ("the
`sdf_from_mesh` empty-mesh GPU fix in the path we are about to port"). The
rest of the list — the builders layer, the LOD bake, the `local_thor.gin`
template — are the valuable half of the harvest and belong to plan items 13
and the local-scheduler work, not to week 1.

## 1 + 2. The `FastNoiseLite.h` compile fixes

**What upstream changed.** `src/infinigen/terrain/source/common/utils/FastNoiseLite.h`
is a header-only rewrite of Auburn/FastNoise that Zeyu Ma turned into free
functions so it can be compiled for both the CPU and the GPU. It used
`assert()` and `NULL` without including `<cassert>`/`<cstddef>`, so it only
compiled when the includer happened to have pulled those in first. The
current includers do (`source/cpu/utils/FastNoiseLite.cpp` and
`source/cuda/utils/FastNoiseLite.cu` both `#include <assert.h>`), which is why
upstream never noticed — until a toolchain stopped providing them
transitively. ebajemon-tw3 hit that with llvm 21 on macOS arm64; DiegoGomesDG
hit the `NULL` half.

**Why it matters here.** We are aarch64 with a different libstdc++ and a
CUDA 13.2 toolchain, and the header is compiled into every terrain element
and surface on both the CPU and the CUDA path. This is a portability fix for
exactly the class of toolchain we are building on, and it costs nothing.

**Why we took it as two commits.** The two fixes are in the same file, from
two different forks, and the plan's wording ("the two `#include` fixes")
covers both readings: the two include lines, and the two upstream-unmerged
fixes to that header. Taking both commits — rather than hand-merging them
into one — keeps authorship and attribution intact, which is the point of
`-x`.

**Deviation (#5, `080eb671`).** Upstream's commit touches **one** of the two
copies of the header in this tree. The second copy,
`src/infinigen/terrain/source/cpu/soil_machine/include/FastNoiseLite.h`, is
the original vendored Auburn/FastNoise C++ class and had the identical
`#include <cmath>`-only shape, so we gave it the same two includes. Be
precise about what this is: **a self-containedness guard, not a live bug
fix** — the soil_machine copy uses neither `std::size_t` nor `assert` today
(it is the class-based original, unlike the terrain copy, which Zeyu Ma
rewrote into free functions). The upstream fix was needed for the terrain
copy, where the rewrite does use both.

## 3. The Boolean EXACT solver — the one that can burn a night

**What upstream changed.** One line in
`src/infinigen/core/constraints/example_solver/populate.py`: the Boolean
modifier that cuts an asset with its cutter asks Blender for the `EXACT`
solver instead of `FAST`. Upstream's reasoning, from the commit message:

> The Blender Boolean modifier's FAST solver (legacy BMesh CSG) can enter an
> infinite loop in C++ when given degenerate or near-degenerate geometry. No
> Python exception is raised; the worker just hangs forever.

with the trigger being sub-microsquare faces on a cutter or a parent surface
(solidified walls intersecting across floors), and the cost being roughly
+20 % wallclock on a typical scene.

**Why we took it.** This is the worst failure mode in the overnight loop: not
a crash, not a slow render, a worker that hangs silently and burns a night.
`populate` is also the stage we are already worried about for a different
reason. A bounded +20 % is worth converting an unbounded silent hang into.

**Deviations.** Two paths in that commit do not exist in this tree, because
the commit predates upstream's move to the `src/` layout and the v2 rewrite
of the changelog:

- `infinigen/core/constraints/example_solver/populate.py` →
  `src/infinigen/core/constraints/example_solver/populate.py`. The pre-image
  blob is byte-identical (`2ee0bdc3`), so the one-line change applied
  unmodified.
- `docs/CHANGELOG.md` (the v1-era changelog, upstream's `v1.19.0` section)
  no longer exists — upstream replaced it with `docs/source/CHANGELOG.md`.
  The upstream changelog note is kept, verbatim in substance, under a new
  `Unreleased` heading in `docs/source/CHANGELOG.md`.

## 4. `sdf_from_mesh` on an empty mesh

**What upstream changed.** `sdf_from_mesh_common` ended with
`assert(min_dist != 1e9)`: on a mesh with no faces (or only degenerate
faces) the assert took the worker down, and in an `NDEBUG` build it returned
a distance computed from an uninitialised `sign`. The fix returns a large
positive SDF (`1e5`) instead, initialises `abs_plane_dist` and `sign`, moves
the sign tolerance `eps3` from `0` to `1e-6` so a near-coplanar triangle
cannot flip the SDF sign, and spells `min`/`abs`/`sqrt` as
`fminf`/`fabsf`/`sqrtf` — the CUDA-safe forms.

**Why we took it.** The scout flagged it as sitting in the CUDA terrain path
we are about to port to sm_110. Note the honest detail: in *this* tree
`sdf_from_mesh` is compiled by the **CPU** translation unit
(`source/cpu/sdf_from_mesh/sdf_from_mesh.cpp`); the header is shared, so the
fix protects whichever path ends up using it. The `eps3` change is the only
behavioural difference beyond the crash fix, and it is a sign tolerance of
1e-6 — flagged in the commit body so nobody is surprised by it.

**Deviation.** Same path move as #3 (`infinigen/terrain/...` →
`src/infinigen/terrain/...`); pre-image blob byte-identical (`dc3fadd`), no
line edited by hand.

## Licence

**BSD 3-Clause, unchanged.** All three source forks ship upstream's own
BSD-3-Clause `LICENSE` verbatim (checked in each fork's tree at the commits
we took: "Copyright (c) 2023, Princeton University"), and this fork is
itself BSD-3-Clause. No relicensing anywhere in the fork network. The
harvest is code only — no textures, no DEM `.npz` assets, no third-party
asset sets — so the "licences are a checked list" rule is not engaged by
anything taken here.

The `FastNoiseLite.h` headers are themselves third-party (Auburn/FastNoise
1.0.1), vendored inside upstream Infinigen and already covered by upstream's
BSD-3 distribution; our change adds two standard-library includes to them.

## Verification

Everything below ran on the Thor in `.venv-harvest/` — a scratch venv built
from the fork's own toolchain (CPython 3.11.13 from `.toolchain/python-3.11`
and the locally built `bpy-4.2.0-cp311-cp311-manylinux_2_39_aarch64.whl`
from `.toolchain/wheels`). The main tree's `.venv`, its installed wheel and
`outputs/sweep` were not touched; the overnight sweep on `tmux last-mile:sweep`
kept running throughout.

### The build still builds here

Every translation unit upstream's `scripts/install/compile_terrain.sh`
builds was compiled with the same flags, into a scratch directory, from a
scratch export of this branch (`git archive HEAD`, with the `infinigen_gpl`
and `glm` submodule headers linked in from their pinned commits so the
worktree's submodule state was left alone):

```
28 CPU objects   (8 elements + 13 surfaces + FastNoiseLite + 4 meshing + sdf_from_mesh + SoilMachine)
22 CUDA libs     (8 elements + 13 surfaces + FastNoiseLite, -arch=native => sm_110)
no compiler diagnostics at all
```

### Each fix, before and after

| Fix | Without it | With it |
|---|---|---|
| `FastNoiseLite.h` stands alone (host) | `g++ -std=c++17`: `'assert' was not declared in this scope` | compiles clean |
| `FastNoiseLite.h` stands alone (device) | `nvcc -std=c++17`: `identifier "assert" is undefined` | compiles clean |
| `NULL` → `nullptr` | 32 bare `NULL` in the terrain copy | 0 |
| `sdf_from_mesh`, empty mesh | aborts: `Assertion 'min_dist != 1e9' failed` (SIGABRT) | returns `1e5`, exit 0 |
| `sdf_from_mesh`, real triangle | — | signed distance ±0.5 as expected |
| `populate.apply_cutter` | `SOLVER=FAST` | `SOLVER=EXACT` |

The "without it" column was produced by checking the pre-harvest files back
out of git (`git checkout 5d1c4455 -- <file>`) and running the same tests,
then restoring them; the failures are the tests themselves, not a manual
inspection.

### The tests

```
PYTHONPATH=src .venv-harvest/bin/python -m pytest \
    tests/infinigen/terrain tests/infinigen/constraints/test_populate_cutter.py -q
9 passed
```

- `tests/infinigen/constraints/test_populate_cutter.py` — the degenerate
  cutter carries a zero-area triangle, a 1e-9 sliver and a duplicated vertex;
  the run happens **in a subprocess with a 120 s timeout**, because the
  failure mode being guarded against is a hang and a hang inside the test
  suite would wedge CI instead of failing it. Reverting the fix turns the
  test's `SOLVER=EXACT` assertion into `SOLVER=FAST`.
- `tests/infinigen/terrain/test_terrain_headers.py` — compiles both copies
  of the header as host code and as device code, checks that neither uses
  `NULL` again, and compiles + runs a program that calls
  `sdf_from_mesh_common` with an empty mesh, with an all-degenerate mesh and
  with a real triangle.

### No collateral damage

The repo's existing suites were run before and after the harvest in the same
scratch venv:

```
tests/infinigen/constraints + tests/infinigen/solver, ignoring the new file
before the harvest: 31 failed, 77 passed, 16 skipped
after  the harvest: 31 failed, 77 passed, 16 skipped
```

Identical. The 31 failures are environmental, not ours: this scratch venv has
no built terrain extension ("Terrain failed import!") and a newer
`python-fcl` than `trimesh` expects (`module 'fcl' has no attribute
'DynamicAABBTreeCollisionManager'`). They fail the same way with the
harvested code reverted.

### Found while building, not fixed

`src/infinigen/terrain/source/cpu/elements/core.cpp` uses `NULL` and
`size_t` with **no `#include` at all**, so it does not compile ("'NULL' was
not declared in this scope" — the compiler even suggests `<cstddef>`). It is
not in upstream's build list, so upstream never compiles it, and our terrain
build does not either. It fails **identically before and after the harvest**,
so this is an upstream latent bug, not something the harvest introduced. It
is a two-line fix (`#include <cstddef>`) and a good candidate for the next
upstream PR — but it is not one of the fixes plan item 6 names, so it is
recorded here rather than quietly taken.

## What was deliberately not taken

- **The rest of the scout's list.** The builders layer (tangos302, ≈1.5k LOC),
  the LOD/Draco bake, `local_apple_silicon.gin` as a `local_thor.gin`
  template, the USD articulated export, the staged re-generation harness.
  All valuable, all in plan item 13 or later, none of them "hours,
  near-zero-risk". This branch is the cheap end only.
- **The other N9 rows** (upstream PRs #506/#507 wall-material kwargs, #512
  Unity export, #493 jellyfish shape keys, #482 Windows int32 assert,
  #485/#486 vlordier's Apple Silicon work, #342 WSL GPU support, #497 GHCR
  publish, #118 AMD/HIP). Each is either out of scope for the Thor, already
  superseded, or belongs with a bigger piece of work.
- **PR #486's OcMesher submodule pointer** in particular: it would move us off
  upstream's OcMesher pin, and our terrain build depends on that pin.
- **Assets of any kind.** The maquette textures (35 MB), the urban DEM `.npz`
  files and the scanned PBR sets are not covered by BSD-3 and need their own
  provenance check.
- **The `fix/populate-cutter-visibility-guard` companion branch** in the same
  fork: it is a visibility guard, not a solver fix, and it is not in plan
  item 6.

## Commits on `thor-harvest` (not pushed)

```
080eb671  terrain: same <cstddef>/<cassert> includes in the soil_machine FastNoiseLite copy
1a4040c9  fix: handle empty meshes in sdf_from_mesh to prevent KeyError on GPU
dc20985e  populate: use EXACT solver for cutter booleans to avoid silent infinite hang
fa987508  fix: failing compilation due to NULL instead of nullptr
a5fd73b8  Fix terrain build on macOS arm64 with llvm 21 (missing cstddef/cassert includes)
fc393bfd  tests: regression tests for the four harvested upstream fixes
134a00df  gitignore: ignore .venv-harvest, the fork-harvest scratch venv
```

The first five are cherry-picks; `git log --format='%an <%ae>'` on them shows
`ebajemon-tw3`, `Diego Davidovich Gomes` (×2) and `Louise Fox` — the
upstream authors, not ours — and each commit body carries the upstream
`(cherry picked from commit <sha>)` line plus a note describing what had to
be reconciled for this tree.

## Reproducing the verification

```bash
# scratch venv (never the main tree's .venv - the sweep is running on it).
# The toolchain and the bpy wheel live in the main tree; only read them.
MAIN=~/src/infinigen-thor
$MAIN/.toolchain/python-3.11/bin/python3.11 -m venv .venv-harvest
.venv-harvest/bin/pip install --no-deps $MAIN/.toolchain/wheels/bpy-4.2.0-*.whl
.venv-harvest/bin/pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.2.2 \
    pandas==2.2.3 opencv-python-headless==4.10.0.84 tqdm gin_config procfunc \
    psutil networkx pyyaml pytest pytest-timeout ruff==0.14.5

# the tests
PYTHONPATH=src .venv-harvest/bin/python -m pytest \
    tests/infinigen/terrain tests/infinigen/constraints/test_populate_cutter.py -q
```

The dependency pins are not upstream's: upstream allows `numpy>=1.25,<2` and
`scikit-learn<1.4`, and the newest `scikit-learn` wheels are built against
the numpy 2 ABI, which does not load under numpy 1.26. `numpy==1.26.4` +
`scikit-learn==1.2.2` is the combination that works on this box.
