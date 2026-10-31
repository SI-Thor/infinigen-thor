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

## The receipts are committed

`receipts/` holds the JSON receipts from the runs that produced the numbers in
`../THOR-MAKING-NOT-CONSUMING.md`. A number without its receipt is a claim.

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
