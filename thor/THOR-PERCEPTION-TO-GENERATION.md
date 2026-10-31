# Making a generative system out of a perception system, on a Thor

**Date:** 2026-10-31 · **Target:** `SI-Thor/infinigen-thor` · **Read with:** `THOR-EXPLOITS.md`, `SPIKE.md`

> *"we know what they optimized this for. but we can treat that just as what it is and optimize real-time world generation from it."*

---

## 0. The thesis in one line

**A perception stack is the analysis half of a pair; generation is the synthesis half; the Thor is already built for the analysis half and is *more* suited to the synthesis half than its own marketing suggests — because diffusion-style generation is GEMM-bound like perception, and the Thor has no RT cores but has FP8/FP4 tensor cores and 273 GB/s of unified bandwidth.**

And the part nobody ships: **NVIDIA's version is open-loop.** You give it a depth map, it generates a video, and nobody checks whether re-perceiving that video recovers the depth you started from. *That* closure is the research, and **Infinigen is the only instrument on earth that can train it** — because Infinigen knows the true posterior over scenes exactly, and no real capture does.

---

## 1. The mathematics

### 1.1 One factorization, two directions

Let `x` be a world (image, video, 3D scene) and `y` what the sensors see. The perception stack is a **measurement operator**:

```
y = A(x) + ε            A = the analysis operator (a neural net: lossy, learned, fixed)
                         ε = sensor noise
```

Perception computes `A`. A generative world system wants to **sample from the fiber**:

```
x̂ ~ p(x | y)     where  supp(p(x|y)) = A⁻¹(y) = { x : A(x) = y }
```

The fiber `A⁻¹(y)` is **the set of worlds that would have produced exactly this percept**. That set is the generative target. Everything below is about learning its geometry cheaply.

The size of that set is the whole game. Formally, the information perception destroyed is

```
I(x ; y) = H(x) − H(x | y)
```

`I(x;y)` is exactly *the information the perception system preserved*. `H(x|y)` is *the freedom the generator has*. **Neither is computable on real data** — you don't know `p(x)` over real scenes. **On Infinigen you do**, because the procedural generator *is* `p(x)`, exactly. So the Thor can measure a quantity that is otherwise unmeasurable. That is §5.

### 1.2 The score-based inverse problem — the exact math

Bayes: `p(x|y) ∝ p(y|x)·p(x)`. Score-based sampling never needs `p(y)`; it needs two gradients:

```
∇_x log p(y|x)      the likelihood score   (how wrong is the measurement?)
∇_x log p(x)        the prior score         (what worlds look like at all?)
```

For a **linear** `A` with Gaussian noise, the likelihood score is closed-form:

```
∇_x log p(y|x) = −Aᵀ(y − A x) / σ²
```

— which is the whole of the plug-and-play / DPS / IRGM inverse-problem machinery, and it is why diffusion models became the tool of choice for inverse imaging (Chung et al., 2022–2023).

For a **nonlinear learned** `A` — a depth network, a segmentation network — the exact score is intractable, but the **Jacobian–vector product** `Aᵀ(y − A x)` is computable by backpropagation through the perception stack itself, and DPS-style surrogates replace `A(x)` by a denoiser's estimate `Â(x)`. This is the *score-based inverse problem* formulation, and it is **precisely** "generate a world that is consistent with what the sensors saw."

**The elegant consequence:** the perception network is not just a conditioner. It is the **likelihood term**. It supplies the gradient that says "your generated world does not match the measurement." That is why one network can serve as conditioner, critic, and loss at once (§4.3).

### 1.3 The sufficiency collapse — why real-time is possible

The expensive part of a diffusion model is the prior: `H(x|y)` may be huge (textures, materials, fine detail — none of it is determined by depth). But look at the split:

| content | determined by `y`? | cost |
|---|---|---|
| **geometry** — layout, occlusion, surfaces, camera | **yes** | deterministic inverse: one forward pass |
| **appearance** — texture, material, lighting detail | **no** | needs the prior: diffusion, iterative |

So the correct architecture is **two-tier**, and this is what makes "real-time world generation" tractable rather than a moonshot:

1. **A deterministic geometric inverse** `D_geo(y)`. A regression, not a sampler. One forward pass, full tensor-core speed, no iteration. Trainable to high fidelity with Infinigen's exact GT.
2. **A stochastic appearance prior** on the residual `x − D_geo(y)`, distilled to a handful of steps (consistency models / shortcut models / flow matching / rectified flow — Song et al. 2023, Lipman et al. 2023, Liu et al. 2023). This is where the expensive math lives, and where the Thor's FP8/FP4 earns its keep.

**If you only ship tier 1, you already have a real-time world generator.** The stochastic tier buys *diversity* — how many worlds are consistent with this percept — which is a scientifically interesting number and a product feature, not a prerequisite for the loop to close.

### 1.4 The residual is the generative target

Two ways to say the same thing:

- Information-theoretically: the generator models `H(x|y)` — the entropy the perception system left behind.
- Casey's way: **the canon is what didn't come through.** FLT-negative-space says *"what did NOT come through IS the signal."* A perception-conditioned generator is a machine that manufactures exactly the part of the world that did not come through.

They are the same mathematics. The negative-space doctrine is the generative residual; the generative residual is the negative space.

---

## 2. The Thor, honestly

### 2.1 What it has, what it does not

| | measured on the box | consequence |
|---|---|---|
| FP8 e4m3 GEMM | **176.96–185.95 TFLOPS** (8192³, 4 runs) | diffusion's denoiser is GEMM-bound → **this is the right machine** |
| FP16 / BF16 | 106.10–113.77 TFLOPS | the "reference" precision |
| FP32 (non-tensor) | 6.37 TFLOPS | never the path |
| bandwidth | 262.8 GB/s measured (273 spec) | the VAE decoder wall |
| unified memory | 128 GB, one pool, `malloc` is device-addressable | **the scene can be enormous** |
| **RT cores** | **NONE** | OptiX is dead; but 3DGS rasterization is fine (sort + alpha blend, no traversal) |
| MIG | 10 TPCs | perception and generation can be isolated |
| tensor cores | 96, 5th-gen | |

### 2.2 The FP4 question — the "measured no" was a *software* limit

EXP C measured FP4 as a no: 13.69 TFLOPS via `torchao`'s NVFP4 path, 1.75× the cost of just dequantising to BF16 and multiplying.

But `qdox-labs/thor-hacks` measured the hardware directly with `ptxas`:

```
tcgen05.mma.sp cta_group::2 kind::mxf4nvf4.block_scale.scale_vec::4X
  -> OK on sm_110a   (no on plain sm_110)
```

**The block-scaled NVFP4 tensor-core path assembles on this GPU — but only with `-arch=sm_110a`.** The "no" came from `torch 2.15 + torchao 0.18` not having a working cast/op path, not from the hardware. NVIDIA's own Cosmos NIM prerequisites say **NVFP4 requires compute capability ≥ 10.0**; the Thor is **11.0**.

So the FP4 question is *open, not closed*, and it is worth exactly one experiment (§6, L0): write a small `tcgen05.mxf4nvf4` GEMM at `-arch=sm_110a` and measure it. If it lands near the marketed density, the generator's step count halves again.

### 2.3 The arithmetic — what actually fits in a frame

**Budget:** 30 fps → 33.3 ms. At 180 TFLOPS FP8 → **6.0 TFLOP per frame-budget.**

**Model cost:** a DiT forward is `≈ 2·N·T` FLOPs per step, `T` = tokens. Take 16 latent frames at 256×256, VAE f8, patch 2 → `T = 16 × 16 × 16 = 4096` tokens.

Attention is not free at 4096 tokens: `4·T²·d` per layer, `d=1024`, 12 blocks → **0.825 TFLOP/step**, which is 25 % of a 300 M model's step and *more than half* of a 100 M model's step. The table includes it.

| params | MLP/step | + attention | 4 steps | at 180 TFLOPS FP8 | fps | MLP-only fps |
|---|---|---|---|---|---|---|
| 100 M | 0.82 TFLOP | 1.64 TFLOP | 6.6 TFLOP | 37 ms | **27** | 55 |
| 300 M | 2.46 TFLOP | 3.29 TFLOP | 13.1 TFLOP | 73 ms | **14** | 18 |
| 1 B | 8.19 TFLOP | 9.02 TFLOP | 36.1 TFLOP | 200 ms | 5.0 | 4.1 |
| 4 B (Cosmos3-Edge) | 32.8 TFLOP | 33.6 TFLOP | 134 TFLOP | 747 ms | 1.3 | 1.1 |

Flash / linear attention changes the calculus: at 4096 tokens the attention term is what stops a small model from being *very* small. If you go to 1024 tokens (8 latent frames at 256²), attention falls 16× and the MLP dominates again.

`thor/probe-thor-gen.sh` measures the real block, not this estimate — use its table, not this one.

**Two conclusions:**

1. **A 100–300 M conditioned generator at 4 steps is real-time on this Thor.** That is your interactive renderer. It is small *because it is conditioned* — the conditioning buys the model size.
2. **A 4 B model at 4 steps is ~1.4 fps at 256².** Cosmos3-Edge is therefore a **research instrument on this box**, not your interactive renderer. Use it to close the loop and to teach; use a small distilled model to serve.

**The bandwidth wall.** Decoding 121 frames at 720p is 225 MB of control media and 75 MB per video cap — the decode and the transfer are the wall, not the GEMMs. **Generate in latent space** (8× spatial, 4× temporal compression ⇒ 1/32 the bytes), and **decode on the laptop**, which is the seat. The Thor never has to move pixels it does not need to.

### 2.4 What "no RT cores" kills, and what it doesn't

Kills: OptiX ray tracing, so Cycles falls back to CUDA (already handled — see `THOR-EXPLOITS.md`).

Does **not** kill: diffusion (no traversal), 3D Gaussian Splatting (sort + alpha blend), epipolar-plane attention, and any VAE/UNet decoder. **The absence of RT cores costs you almost nothing in this particular project**, because a generative world model is not a ray tracer. It costs you *ground-truth render speed* — which is offline anyway.

---

## 3. What NVIDIA already shipped — and the gap next to it

**Cosmos3-Edge (4B)** — released Jul 2026, listed as running on **Jetson AGX Orin / Thor / RTX PRO 6000**, for *"edge deployment, real-time robotic policy, and visual reasoning."* The Cosmos NIM docs say that **on Jetson AGX Thor it applies a 70 % unified-memory share automatically.**

Its **Transfer** controls are exactly perception outputs: `edge`, `blur`, `depth`, `seg`, `wsm` (world-space-map). And the docs are explicit:

> *"The server cannot derive depth, segmentation, or WSM controls. Supply a precomputed control video."*

**So NVIDIA's own architecture requires you to compute the perception locally and hand it to the generator.** That *is* "a generative system out of a perception system" — shipped, on your box.

**The gap: it is open-loop.**

```
NVIDIA's loop:     y  ──▶  G  ──▶  x̂            (generate; nobody checks)
The closed loop:   y  ──▶  G  ──▶  x̂  ──▶  A  ──▶  ŷ   (and ŷ must equal y)
```

Define the **round-trip operator** `R = A ∘ G`. A perfect generator has `R = identity` on the observable subspace. Nobody ships the loss that enforces it, because for real video data you cannot: there is no way to know whether `ŷ` *should* have equalled `y`. **Infinigen is the exception.** With exact ground truth you can compute

```
L_self    = ‖ A(G(y)) − y ‖              self-consistency: no ground truth needed
L_percept = ‖ A(G(y)) − A(x*) ‖          the ideal objective — requires GT
L_appear  = ‖ G(y) − x* ‖                full fidelity — requires GT
```

and measure the gap between `L_self` and `L_percept`. **That gap is a research result**: how much does a perception-conditioned generator actually preserve, versus how much it *appears* to preserve by the only test anyone currently runs?

---

## 4. The three roles of one network

### 4.1 Conditioner
`G(y)` needs `y`. The perception net produces it.

### 4.2 Likelihood term
§1.2: the perception net supplies `∇_x log p(y|x)`. It is the thing that says "your generated world does not match the measurement."

### 4.3 Critic — and this unblocks a workstream of yours

Your `builder-ide` core-spec says:

> *"a critic enters service only inside its band"* — a calibration record of **≥ 30 seeded-defect reference renders, measured FP/FN, target band**; until a critic qualifies, automated verdicts are **scripted checks labelled `verdict@scripted`**.

**You currently have no critic with a band.** And a neural perception stack is *exactly* a critic with a band: it has measurable FP/FN on a defect set, it is deterministic, it is fast, and — crucially — **Infinigen can manufacture the calibration set.** `Prediction.render_spec {rung, resolution, seed}` already gives you reproducible, seeded renders. That is the ≥30 seeded-defect reference set, generated on demand.

So: **the perception net is the critic that was blocking your critic protocol.** It is mechanical, re-scorable against the same inputs (your sealed payload already carries the spec), and its defect list is exactly "verdict = Critic, defects" — depth mismatch, missing object, occlusion error.

That is a *concrete unblock of a blocked workstream*, using an asset you already have, and it is independent of the generative half.

### 4.4 The witness chain, made of pixels
Your three-record rule: *prediction = builder, sealed; verdict = Critic, defects.* The closed loop adds the fourth: **`R(x̂) = y` is the receipt.** A generated world that does not round-trip through perception is not a world — it is a picture of one.

---

## 5. The measurement only Infinigen can make

`I(x ; y) = H(x) − H(x|y)` — the information a perception system preserved about a scene.

Nobody can measure this on real data: you do not know `p(x)` over real scenes. **On Infinigen you do** — the procedural generator *is* `p(x)`, exactly, and you can sample it at will. So:

- Estimate `I(x;y)` between scenes and their perception, with known `p(x)`.
- The result is **"how much does a perception system destroy?"** — a number that sizes the entire generative design, and that nobody has.
- And it is measurable *per-modality*: depth preserves X, segmentation preserves Y, optical flow preserves Z. That tells you **which conditioner to use**, and therefore which model size you can afford.

**This is the single most novel thing your particular asset stack enables, and it needs no new GPU capability.** It needs the Thor to run the perception net and the Infinigen renders to supply the truth.

---

## 6. The experiment ladder

Cost is in **Thor-days** (one person, one box, GPU-hours included).

| # | experiment | cost | what it decides |
|---|---|---|---|
| **L0** | **Measure the real budget.** FP8 GEMM at `sm_110`; FP4 `tcgen05` at `sm_110a` (does the block-scaled path *run*, not just assemble?); a DiT-block forward at 4096 tokens; a VAE-decode-shaped conv. → `thor/probe-thor-gen.sh` | **0.25 d** | the model size, the step count, the resolution. Everything else depends on this. |
| **L1** | **Get a perception net on the box.** None exists yet. A small depth+seg model (Depth-Anything-V2-Small-class). Measure its FP8 throughput on the Thor. | **0.5 d** | whether the conditioner is cheap enough to always be resident |
| **L2** | **Run Cosmos3-Edge 4B on the Thor.** Feed it Infinigen's GT depth/seg as the control. Measure real fps at 256²/480². | **1–2 d** | whether NVIDIA's on-device tier is usable here, and at what resolution |
| **L3** | **Close the loop, first version.** `R = A ∘ G`. Compute `‖A(G(y)) − y‖` over a sequence. Plot the drift per frame. | **1 d** | **the first real result of the project.** Does perception-conditioned generation actually round-trip? |
| **L4** | **Measure `I(x;y)`** on the Infinigen distribution, per modality. | **2–3 d** | how much freedom the generator has, and which conditioner to buy |
| **L5** | **Train the real-time generator.** 100–300 M, conditioned on depth+seg+flow, consistency/shortcut-distilled to 4 steps, FP8, latent space. Trained on Infinigen GT. | **2–3 wks** | the interactive renderer: 30 fps at 256² on the Thor |
| **L6** | **3DGS world state.** The generator emits Gaussians; rasterize (no RT cores needed); perception net as critic; Infinigen supervises exactly. | **3–4 wks** | **view consistency for free** — the thing 2D video generation cannot do |
| **L7** | **MIG partition.** TPCs 0–4 generate, 5–9 critique, concurrently. | **1 wk** | whether the loop can run truly concurrently |

**Run L0 → L3 first.** That is 3 days and it produces the number that decides everything else. L4 is the paper.

---

## 7. What could kill it — stated plainly

1. **Weight-swap thrash is your own diagnosed constraint** ("the hardware's real constraint," per your own research). Perception net + generator + VAE + Cycles cannot all be resident. **Design for one resident model at a time**, or use MIG (10 TPCs). The closed loop may be *serial by nature* — generate, then perceive — which halves your throughput and is fine for research but not for interactive.
2. **Cosmos3-Edge's actual Thor performance is unpublished.** Nobody has measured it. L2 is a genuine unknown, not a formality.
3. **NVFP4 may be a software dead end** even if `ptxas` assembles it. `torch 2.15 + torchao 0.18` has no working path; you may have to write the GEMM yourself (CUTLASS with `CUTLASS_NVCC_ARCHS=110a`, ≥ 4.2.0, per `thor-hacks`). That is a week of work for a 2× on top of a good number.
4. **The deterministic inverse may be good enough**, making the stochastic tier unnecessary for the product. That would be a *win*, not a loss — but it would make the diffusion half a science project rather than a product component. Plan for both.
5. **The `.cpu()` trap still applies.** Any tensor pulled into pageable memory runs at 0.34 GB/s and never fails — it just takes eleven seconds. The closed loop will want to read the generated frame back to compare it. **Do not.** Use the unified pool: `cupy.cuda.UnownedMemory` / `wp.array(ptr=..., copy=False)`.

---

## 8. The two-tier summary

```
   Infinigen (p(x) exact)                Thor (128 GB unified, 180 TFLOPS FP8)
   ─────────────────────                 ──────────────────────────────────────
   world x* ──render──▶ RGB + exact      ┌──────────────────────────────────┐
                      depth/seg/flow     │  perception net A                │
        │                                │   (conditioner + likelihood +    │
        │                                │    critic — ONE network, 3 jobs) │
        ▼                                └──────────────────────────────────┘
   perception net A(x*) = y  (with its    ┌──────────────────────────────────┐
   real errors — the exact pairing)       │  generator G                     │
        │                                │   tier 1: deterministic geometry │
        ▼                                │   tier 2: distilled prior (4 stp)│
   dataset: (y, x*) with EXACT           └──────────────────────────────────┘
   supervision                                     │
                                                     ▼
                                          A(G(y)) =? y
                                          the closed loop nobody ships
```

---

## 9. Sources

- NVIDIA/cosmos README (Jul 2026) — Cosmos3-Edge 4B, "Jetson AGX Orin / Thor / RTX PRO 6000", edge/real-time tier.
- Cosmos NIM docs (4.0.0) — Transfer controls `edge|blur|depth|seg|wsm`; "the server cannot derive depth, segmentation, or WSM controls"; compute capability floors (BF16 ≥ 8.0, FP8 ≥ 8.9, **NVFP4 ≥ 10.0**); **"On NVIDIA DGX Spark or Jetson AGX Thor, the NIM applies a 70 % unified-memory share automatically"**; validated on aarch64.
- `build.nvidia.com` Cosmos-Transfer2.5-2B model card — 65.4 GB VRAM; H100 PCIe 870 s for a 720p/121-frame segmentation transfer (**the reason the 2B Transfer tier is not real-time, and the reason the 4B Edge tier exists**).
- `qdox-labs/thor-hacks` — `tcgen05`/`mxf4nvf4` block-scale **OK on sm_110a**; sm_110→sm_110 renumber; PTX 9.0 floor; sm_87 silent failure; measured 262.8 GB/s.
- `builder-ide/docs/lab/exp-{fp8-tensorcores,unified-memory,warp-terrain}.md` — FP8 1.57–1.75× FP16; FP4 measured no *via torch/torchao*; unified memory one-pool; Warp native sm_110 SASS.
- `builder-ide/docs/design/core-spec.md` — the critic band / `verdict@scripted` / three-record rule / `Prediction.render_spec` / weight-swap thrash.
- `src/infinigen2/exporters/render_cycles.py` — `RENDER_CYCLES_GT_PASS_TYPES` (DEPTH, SURFACE_NORMAL, OPTICAL_FLOW, OBJECT_INDEX) — **the exact supervision, already implemented.**
- Method: DPS / IRGM (Chung et al. 2022–2023) for score-based inverse problems; consistency models (Song et al. 2023), flow matching (Lipman et al. 2023), rectified flow (Liu et al. 2023) for few-step sampling; 3DGS (Kerbl et al. 2023) for the editable differentiable world state; DiT (Peebles & Xie 2023) for the architecture.

*Every hardware number above is either measured by this team on the box or cited from an NVIDIA primary source. The FLOP tables are derived, and the derivation is shown so you can check it.*
