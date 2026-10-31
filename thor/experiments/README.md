# Experiments: making perception and generation two sides of one coin

Every script here **runs with nothing but numpy**. No torch, no GPU, no
download, no dataset. That is deliberate: the claims are about *mathematics*,
and mathematics that needs a GPU is not yet a claim.

Run any of them with `--selftest` / `--help` and read the receipt it writes.

| script | what it measures | runnable now? |
|---|---|---|
| **`toy_coin.py`** | **C1–C4.** A score field is a generator *and* a critic; the naive closed-loop objective `‖A(G(y))−y‖` is **degenerate** — it certifies physically impossible worlds. | ✅ numpy only |
| **`fiber_probe.py`** | The **fiber width**: how much of the world the perception stack actually determines. Laplace approximation, checked against an **analytic answer**. Gives the per-frame schedule rule. | ✅ numpy only |
| **`distill_closed_loop.py`** | **E2.** Distil in the closed-loop metric instead of in pixels. Three students, one budget: pixel distillation, consistency-only, consistency + plausibility. | ✅ numpy only |
| **`closed_loop.py`** | **E1/E3.** The production harness: measures `R = A∘G` on real Infinigen GT, with a positive control (identity ⇒ residual exactly 0) and a negative control (identity+ε ⇒ residual must grow). | ✅ self-test now; real runs need the Thor |
| **`test_closed_loop.py`** | End-to-end proof that the harness works on a **real Infinigen output directory** — the layout is built from `format.py`, not assumed. | ✅ numpy only |
| **`test_hardware_info.py`** | GPU detection must not lose a GPU to an unsupported sibling field. Fake `nvidia-smi` on PATH; no GPU, no Jetson, no driver. | ✅ numpy only |

## The receipts are committed

`receipts/` holds the JSON receipts from the runs that produced the numbers in
`../THOR-MAKING-NOT-CONSUMING.md`. A number without its receipt is a claim.

## The loader was wrong, and the end-to-end test is why

`closed_loop.py`'s first loader assumed `rgb_*.png` and `*obj_index*.npy` at
the top level. The **real** layout, from
`src/infinigen2/exporters/util/format.py`, is:

```
<output>/<camera>/0001.png                the main image   ("%c/%f.png")
<output>/<camera>/depth_0001.npy          ("%c/depth_%f.npy")
<output>/<camera>/optical-flow_0001.npy   ("%c/optical-flow_%f.npy")
<output>/<camera>/object_0001.npy         ("%c/object_%f.npy")
```

with `%c` = the camera name (**a subdirectory**) and `%f` = `f"{frame:04d}"`.
The assumed loader found **nothing** on a real output and reported *"no frames
found"* — which reads like a data problem and is actually a loader problem.

`test_closed_loop.py` builds a directory with the real layout and asserts the
loader finds it. Against the old loader: **4 of 5 tests fail** (found 0 frames;
the object-index metric returns 0.0 for two different label maps; the CLI exits
2). The fifth — that a non-Infinigen directory yields no frames — passes
against both, because it should.

**Three more defects the end-to-end test caught:**

- **The object index is a LABEL MAP** `(H,W) uint32`, not one-hot, so
  `argmax(-1)` compared the wrong thing and returned 0.0 for two *different*
  maps. Comparing the labels is correct.
- **Every metric must be an ERROR, lower-is-better.** The RGB metric was a
  PSNR — the other direction — so the control *"identity must give exactly 0"*
  was unsatisfiable by construction (identity gave 168 dB). It is now RMSE.
- `read_any` was casting everything to float32, destroying the uint32/uint8
  types; the metrics cast for themselves instead.
- `NoisyGenerator` perturbed only *floating* arrays, so once the label maps
  kept their integer dtype the bad generator left them bit-exact — and the
  negative control reported *"the noise changed nothing"* for a plainly
  broken generator. A bad generator corrupts the labels too.

## The controls are the point

Each script carries a control that **can fail**, and each of them did fail
during development, which is how the bugs were found:

- `toy_coin.py` — a negative control that must show the deterministic inverse
  is *consistent* (round-trip at the noise floor) while being *implausible*
  (far off the manifold). It cannot be satisfied by a metric that only looks at
  one of the two.
- `fiber_probe.py` — the Laplace covariance and fiber width are compared
  against a **closed-form answer**; a sharper prior must give a narrower fiber;
  the sampler must reproduce the analytic conditional mean (0.014) and spread
  (0.9%).
- `distill_closed_loop.py` — the pixel student and the consistency-only
  student must **agree** (both are the blur). If they disagree, the experiment
  is not measuring what it claims.
- `closed_loop.py` — identity generator ⇒ residual exactly 0; identity+ε ⇒
  residual must grow. If either fails, the harness aborts and reports no
  numbers.

## The shape on the Thor

```
python3 closed_loop.py --source /path/to/render_out \
    --perception mypkg.perception:build --generator mypkg.generator:build \
    --out runs/closed-loop-01
```

`--perception` / `--generator` / `--plausibility` take `module:factory`. The
factory returns a callable:

```python
class Perception:
    name = "..."
    def perceive(self, frames: list[dict]) -> list[dict]: ...

class Generator:
    name = "..."
    def generate(self, obs: list[dict], history=None) -> list[dict]: ...
```

Frames are dicts of numpy arrays keyed by `rgb`, `depth`, `flow`,
`obj_index`, `semantic_seg` — which is exactly what Infinigen writes when
`RENDER_CYCLES_GT_PASS_TYPES` includes `DEPTH`, `OPTICAL_FLOW` and
`OBJECT_INDEX`.
