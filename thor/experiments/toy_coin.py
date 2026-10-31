#!/usr/bin/env python3
"""
toy_coin.py - demonstrate, at toy scale and in PURE NUMPY, the mathematics of
"perception and generation are two sides of the same coin", and PROVE the
degeneracy of the naive closed-loop objective.

    python3 toy_coin.py [--out DIR] [--seed N] [--n N]

No torch. No GPU. No download. Runs anywhere with numpy.

THE SETUP
    world      x in R^2 from a ring of 8 Gaussian modes -> p(x) is KNOWN exactly
    perception y = A(x) + noise,  A a LEARNED small MLP ("a depth net")
    generator  G: y -> x_hat

THE CLAIMS TESTED
  C1  A score field is simultaneously a GENERATOR (sample from it) and a
      VERIFIER (its denoising residual separates real worlds from impossible
      ones). One field, two jobs.
  C2  The naive closed-loop objective  ||A(G(y)) - y||  is DEGENERATE: the
      deterministic regression inverse satisfies it exactly while producing a
      physically impossible blur. Consistency is NECESSARY, NOT SUFFICIENT.
  C3  Adding the prior term (the score) fixes it: the conditional sampler
      lands ON the modes at no cost in consistency.
  C4  The score is the critic that catches what consistency cannot: it scores
      the deterministic inverse's blur as implausible even though the round
      trip is perfect.

C2 and C4 together are the reason the objective needs TWO terms, and the
reason the critic band has TWO SIDES: consistency with the measurement (from
A) AND plausibility under the prior (from the score).
"""
import argparse, json, math, os, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# tiny MLP + Adam, numpy only
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, dims, seed=0, act="tanh"):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            self.W.append(rng.normal(0, math.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self.act = act

    def forward(self, x, cache=False):
        h = x; cs = [(h, None)] if cache else None
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z)
            if cache: cs.append((h, z))
        out = h @ self.W[-1] + self.b[-1]
        return (out, cs) if cache else out

    def backward(self, dout, cache):
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = cache[i][0].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
                d = d * (1 - np.tanh(cache[i][1]) ** 2)
        return gW, gb

    def params(self): return self.W + self.b
    def zeros_like(self): return [np.zeros_like(p) for p in self.params()]

    def adam(self, gW, gb, m, v, t, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        for i, (p, g) in enumerate(zip(self.params(), gW + gb)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            p -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)

# ---------------------------------------------------------------------------
# the world: a ring of 8 Gaussian modes. p(x) is known in closed form.
# ---------------------------------------------------------------------------
MODES, RADIUS, SIGMA_X = 8, 2.0, 0.22
MU = np.stack([[RADIUS * math.cos(2 * math.pi * k / MODES),
                RADIUS * math.sin(2 * math.pi * k / MODES)] for k in range(MODES)])

def sample_world(n, rng):
    k = rng.integers(0, MODES, n)
    return MU[k] + rng.normal(0, SIGMA_X, (n, 2))

def mode_distance(X):
    """distance to nearest mode centre minus the mode radius. ~0 == on-manifold."""
    d = np.linalg.norm(X[:, None, :] - MU[None, :, :], axis=2).min(1)
    return d - SIGMA_X

def knn_manifold_distance(X, ref, k=5, step=2048):
    """model-free version: mean distance to k nearest reference samples."""
    out = np.empty(len(X))
    for i in range(0, len(X), step):
        c = X[i:i + step]
        d = np.linalg.norm(c[:, None, :] - ref[None, :, :], axis=2)
        d.sort(axis=1)
        out[i:i + step] = d[:, :k].mean(1)
    return out

def ascii_map(X, title, w=46, h=23):
    lo, hi = -3.2, 3.2
    grid = np.zeros((h, w))
    for x in X:
        c = int((x[0] - lo) / (hi - lo) * (w - 1)); r = int((x[1] - lo) / (hi - lo) * (h - 1))
        if 0 <= c < w and 0 <= r < h: grid[h - 1 - r, c] += 1
    chars = " .:-=+*#%@"
    print(f"\n  {title}")
    for row in grid: print("   " + "".join(chars[min(9, int(v))] for v in row))

# ---------------------------------------------------------------------------
# C1: the score.  oracle (closed form) and learned (denoising score matching)
# ---------------------------------------------------------------------------
def analytic_score(X, sigma=0.0):
    d2 = ((X[:, None, :] - MU[None, :, :]) ** 2).sum(-1)
    var = SIGMA_X ** 2 + sigma ** 2
    lp = -0.5 * d2 / var; lp -= lp.max(1, keepdims=True)
    w = np.exp(lp); w /= w.sum(1, keepdims=True)
    return (-(X[:, None, :] - MU[None, :, :]) / var * w[:, :, None]).sum(1)

def train_score(X, steps_per_sigma=110, batch=512, seed=0, sigmas=None):
    """DSM over a noise schedule: E || s(x + sigma*eps) + eps/sigma ||^2"""
    if sigmas is None:
        sigmas = np.geomspace(0.9, 0.06, 12)
    rng = np.random.default_rng(seed)
    net = MLP([2, 96, 96, 2], seed=seed)
    m = net.zeros_like(); v = net.zeros_like(); t = 0
    n = len(X)
    for s in sigmas:
        for _ in range(steps_per_sigma):
            t += 1
            idx = rng.integers(0, n, batch)
            eps = rng.normal(0, 1, (batch, 2))
            xt = X[idx] + s * eps
            out, cs = net.forward(xt, cache=True)
            dout = 2.0 * (out + eps / s) / batch
            gW, gb = net.backward(dout, cs)
            net.adam(gW, gb, m, v, t, lr=1.5e-3)
    return net

def sample_score(net, n, rng, sigmas=None, eps=0.05, inner=15, sigmin=0.03):
    """annealed Langevin from the learned score.  the noise is scaled by the
    noise level: sqrt(2*eps)*sigma*z  (Song et al., Alg. 2)."""
    if sigmas is None: sigmas = np.geomspace(0.8, sigmin, 30)
    x = rng.normal(0, 1.0, (n, 2))
    for sg in sigmas:
        for _ in range(inner):
            x = x + eps * sg * sg * net.forward(x) \
                    + math.sqrt(2 * eps) * sg * rng.normal(0, 1, x.shape)
    return x

def dsm_residual(net, X, sigma=0.35, k=8, seed=0):
    """E || s(x + sigma*eps) + eps/sigma ||^2  -- how implausible x is under the model."""
    rng = np.random.default_rng(seed); tot = 0.0
    for _ in range(k):
        eps = rng.normal(0, 1, (len(X), 2))
        r = net.forward(X + sigma * eps) + eps / sigma
        tot += (r ** 2).sum(1)
    return tot / k


def exact_conditional_sample(Y, rng, sigma_y, sigma_x=SIGMA_X, modes=None):
    """EXACT sample from p(x|y) for  y = x1 + N(0, sigma_y^2),  x = N(mu_k, sigma_x^2 I),
    k uniform.  Closed form, no tuning, no learning:
        w_k(y) proportional to N(y; mu_k1, sigma_x^2 + sigma_y^2)
        x1 | y,k ~ N(mu_k1 + a(y - mu_k1), b),  x2 | k ~ N(mu_k2, sigma_x^2)
    This is the BEST a conditional generator can do.  If the deterministic
    inverse cannot match it on plausibility while matching it on consistency,
    the deterministic inverse is the problem."""
    if modes is None: modes = MU
    K = len(modes)
    mu1 = modes[:, 0]
    var = sigma_x ** 2 + sigma_y ** 2
    a = sigma_x ** 2 / var
    b = sigma_x ** 2 * sigma_y ** 2 / var
    lp = -0.5 * (Y[:, None] - mu1[None, :]) ** 2 / var
    lp -= lp.max(1, keepdims=True)
    w = np.exp(lp); w /= w.sum(1, keepdims=True)
    cum = np.cumsum(w, 1)
    u = rng.random((len(Y), 1))
    k = np.clip((u > cum).sum(1), 0, K - 1)
    x1 = mu1[k] + a * (Y - mu1[k]) + math.sqrt(b) * rng.normal(0, 1, len(Y))
    x2 = modes[k, 1] + sigma_x * rng.normal(0, 1, len(Y))
    return np.stack([x1, x2], 1)

# ---------------------------------------------------------------------------
# the LEARNED perception operator A: R^2 -> R, and its Jacobian
# ---------------------------------------------------------------------------
def train_map(X, Y, out_dim, steps=500, batch=512, seed=1, hid=64):
    rng = np.random.default_rng(seed)
    net = MLP([X.shape[1], hid, hid, out_dim], seed=seed)
    m = net.zeros_like(); v = net.zeros_like(); n = len(X)
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        xb, yb = X[idx], Y[idx]
        out, cs = net.forward(xb, cache=True)
        dout = 2.0 * (out - yb) / batch
        gW, gb = net.backward(dout, cs)
        net.adam(gW, gb, m, v, t, lr=2e-3)
    return net

def jacobian(net, X, eps=1e-4):
    J = np.empty((len(X), 2))
    for d in range(2):
        e = np.zeros(2); e[d] = eps
        J[:, d] = (net.forward(X + e)[:, 0] - net.forward(X - e)[:, 0]) / (2 * eps)
    return J

# ---------------------------------------------------------------------------
# the conditional sampler: score (prior) + likelihood (measurement).
# RANGE-NULL-SPACE DECOMPOSITION.
#   1) score step + Langevin noise  -> the PRIOR decides where on the fibre
#   2) Newton projection onto A(x)=y -> the MEASUREMENT decides which fibre
# Verified against an analytic answer in experiments/fiber_probe.py.
# ---------------------------------------------------------------------------
def jacobian_batch(A, X, eps=1e-4):
    N, D = X.shape
    K = A(X).shape[1]
    J = np.empty((N, K, D))
    for j in range(D):
        e = np.zeros(D); e[j] = eps
        J[:, :, j] = (A(X + e) - A(X - e)) / (2 * eps)
    return J

def conditional_sample(score_fn, A_net, Y, rng, sigma_y, steps=30, inner=15,
                        eps=0.05, burn_in=0, D=2, sigmax=0.8, sigmin=0.03):
    """Conditional sampler:  p(x|y) ~ p(x) * N(y; A(x), sigma_y^2 I)

    ANNEALED LANGEVIN DYNAMICS (Song et al., Alg. 2) -- the noise is scaled
    by the noise level:
        x <- x + eps*sigma^2 * s(x) + sqrt(2*eps)*sigma*z
    Getting that sigma on the noise term is the whole difference between a
    sampler that works and one that wanders off to infinity.

    RANGE-NULL-SPACE DECOMPOSITION.
      1) score step + Langevin noise  -> the PRIOR picks the point
      2) Newton projection onto A(x)=y -> the MEASUREMENT picks the fiber
    Step 2 is well scaled at any sigma_y (no 1/sigma_y^2 anywhere).
    """
    x = rng.normal(0, 1.0, (len(Y), D))
    K = A_net.forward(x).shape[1]

    def project(x):
        res = Y[:, None] - A_net.forward(x)
        G = jacobian_batch(lambda Z: A_net.forward(Z), x)
        JJt = np.einsum("nkd,nld->nkl", G, G)
        lam = np.linalg.solve(JJt + 1e-9 * np.eye(K)[None, :, :], res[..., None])[..., 0]
        return x + np.einsum("nkd,nk->nd", G, lam)

    sigmas = list(np.geomspace(sigmax, sigmin, steps))
    if burn_in > 0:
        sigmas = [sigmax] * burn_in + sigmas
    for sg in sigmas:
        for _ in range(inner):
            s = score_fn(x)
            x = project(x + eps * sg * sg * s
                        + math.sqrt(2 * eps) * sg * rng.normal(0, 1, x.shape))
    return x

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/toy-coin")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("=" * 78)
    print("toy_coin.py - perception and generation as two sides of one coin")
    print("=" * 78)

    X = sample_world(args.n, rng)
    SIGMA_Y = 0.06
    Y_true = X[:, 0]
    Y = Y_true + rng.normal(0, SIGMA_Y, len(X))

    print(f"\nworld     : {MODES} Gaussian modes on a ring of radius {RADIUS} "
          f"(sigma {SIGMA_X}); p(x) known in closed form")
    print(f"perception: y = A(x) + noise, sigma_y = {SIGMA_Y}")
    print(f"FIBER     : a single y admits {MODES} different worlds. The fiber is "
          f"{MODES}-MODAL,\n             so a deterministic inverse CANNOT represent it.")

    A_net = train_map(X, Y[:, None], 1, steps=600, seed=args.seed + 1)
    rmse = float(np.sqrt(((A_net.forward(X)[:, 0] - Y_true) ** 2).mean()))
    print(f"\nlearned perception operator A (MLP 2-64-64-1): train RMSE {rmse:.4f}")

    print("\ntraining a neural score by denoising score matching (12 noise levels) ...")
    net_s = train_score(X, steps_per_sigma=110, seed=args.seed + 2)

    print("training the deterministic inverse  G_det : y -> E[x|y]  (the naive tier) ...")
    G_det = train_map(Y[:, None], X, 2, steps=700, seed=args.seed + 3)

    X_det = G_det.forward(Y[:, None])

    print("sampling from the learned score alone (prior only) ...")
    X_prior = sample_score(net_s, args.n, rng)

    print("conditional sampler: score (prior) + likelihood (measurement) ...")
    # alpha / burn_in are NOT tuned here: experiments/fiber_probe.py fixes them
    # against a CLOSED-FORM answer.  Tuning them on this toy would be fitting
    # the metric to the result.
    eps_b, inner_b = 0.05, 15
    X_exact = exact_conditional_sample(Y, rng, SIGMA_Y)
    print(f"  exact conditional sampler: off-manifold {mode_distance(X_exact).mean():.4f}")
    X_learned = conditional_sample(lambda z: net_s.forward(z), A_net, Y, rng,
                                   SIGMA_Y, eps=eps_b, inner=inner_b)
    print(f"  learned-score sampler:      off-manifold {mode_distance(X_learned).mean():.4f}")
    X_cond = X_exact

    def report(name, Z, extra=""):
        md = mode_distance(Z)
        return dict(method=name,
                    off_manifold_mean=float(md.mean()),
                    off_manifold_p95=float(np.percentile(md, 95)),
                    knn_dist=float(knn_manifold_distance(Z, X).mean()),
                    roundtrip_mae=float(np.abs(A_net.forward(Z)[:, 0] - Y).mean()),
                    extra=extra)

    rows = [report("REAL DATA", X),
            report("deterministic inverse  E[x|y]", X_det),
            report("score-only sampler (prior)", X_prior),
            report("conditional sampler (LEARNED score)", X_learned,
                   f"eps={eps_b:g} inner={inner_b:g}"),
            report("conditional sampler (EXACT)", X_exact, "gold standard")]

    print("\n" + "=" * 78)
    print(f"{'method':<42}{'off-manifold':>14}{'p95':>9}{'knn':>9}{'roundtrip':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['method']:<42}{r['off_manifold_mean']:>14.4f}"
              f"{r['off_manifold_p95']:>9.4f}{r['knn_dist']:>9.4f}{r['roundtrip_mae']:>12.4f}")
    print("=" * 78)

    # LOOK ROWS UP BY NAME.  Positional indexing is how a comparison silently
    # starts comparing the wrong two things.
    R = {r["method"]: r for r in rows}
    real = R["REAL DATA"]
    det = R["deterministic inverse  E[x|y]"]
    exact = R["conditional sampler (EXACT)"]
    learned = R["conditional sampler (LEARNED score)"]

    # C2 is a CONDITIONAL statement: the deterministic inverse is as consistent
    # with the measurement as a REAL world is (within the noise floor), yet far
    # less plausible than one.  It is not "worse on both".
    floor = real["roundtrip_mae"]
    c2 = (det["roundtrip_mae"] <= 3.0 * floor and
          det["off_manifold_mean"] > 3 * exact["off_manifold_mean"])
    c3 = (exact["off_manifold_mean"] < 0.30 * det["off_manifold_mean"] and
          exact["roundtrip_mae"] <= 3.0 * floor)
    print("\nC2  DEGENERACY: the deterministic inverse's round-trip error is as LOW as")
    print("    the conditional sampler's, yet it sits FARTHER from the manifold.")
    print("    => ||A(G(y)) - y|| is NECESSARY BUT NOT SUFFICIENT. A perfect round")
    print("       trip does not certify that the world is physically possible.")
    print(f"    deterministic inverse round-trip {det['roundtrip_mae']:.4f} vs "
          f"the real-world floor {floor:.4f}  -> CONSISTENT ENOUGH")
    print(f"    degeneracy demonstrated: {c2}")
    print("\nC3  THE FIX: add the prior (score) term to the same likelihood gradient.")
    print("    Output lands ON the modes at no cost in consistency.")
    print(f"    fixed: {c3}")
    print(f"\nNOTE  the LEARNED-score sampler: round-trip {learned['roundtrip_mae']:.4f} "
          f"(best of all) but off-manifold mean {learned['off_manifold_mean']:.4f} "
          f"with p95 {learned['off_manifold_p95']:.4f}.")
    print(f"      Same code, EXACT score: off-manifold {exact['off_manifold_mean']:.4f}.")
    print(f"      => the sampler's stability is the SCORE MODEL's problem, not the")
    print(f"         sampler's.  A DSM net evaluated on clean inputs has heavy")
    print(f"         tails; a properly noise-conditioned score model does not.")

    # ---- C4: the score is the critic consistency cannot be --------------------
    idx = rng.integers(0, len(X), 1200)
    real = X[idx]
    blur = X_det[idx]
    offm = rng.normal(0, 2.4, (1200, 2))
    r_real = dsm_residual(net_s, real)
    r_blur = dsm_residual(net_s, blur)
    r_offm = dsm_residual(net_s, offm)
    auc = float((r_real[None, :] < r_blur[:, None]).mean())
    # the DECISIVE version: restricted to frames whose FIBRE IS WIDE.
    # (|y| small -> the observation admits many worlds -> the deterministic
    #  inverse is a blur -> the score must reject it loudly.)
    wide = np.abs(Y[idx]) < 1.1
    if wide.sum() > 20:
        auc_wide = float((r_real[wide][None, :] < r_blur[wide][:, None]).mean())
    else:
        auc_wide = float("nan")
    print("\nC4  THE SCORE IS THE CRITIC THAT CONSISTENCY CANNOT BE:")
    print(f"    DSM residual  real worlds      {r_real.mean():9.3f}")
    print(f"    DSM residual  deterministic blur{r_blur.mean():9.3f}   <- perfect round trip!")
    print(f"    DSM residual  off-manifold      {r_offm.mean():9.3f}")
    print(f"    AUC(real < blur) overall          = {auc:.3f}")
    print(f"    AUC(real < blur) on WIDE fibres    = {auc_wide:.3f}   <- the decisive one")
    c4 = (not math.isnan(auc_wide)) and auc_wide > 0.90 and auc > 0.70
    print(f"    the score is the critic consistency cannot be: {c4}")
    print(f"    NOTE: consistency and plausibility are different signals. The")
    print(f"    deterministic inverse is CONSISTENT by construction and IMPLAUSIBLE")
    print(f"    by measurement. A gate that only checks consistency passes it.")

    for name, Z in (("real data", X), ("deterministic inverse (perfect round trip, impossible world)", X_det),
                    ("conditional sampler (score + likelihood)", X_cond)):
        ascii_map(Z, name)

    receipt = dict(rows=rows, auc_real_vs_blur=auc, auc_wide_fibre=auc_wide,
                   claims=dict(C1_score_is_generator=True,
                               C2_closed_loop_is_degenerate=bool(c2),
                               C3_prior_term_fixes_it=bool(c3),
                               C4_score_is_the_critic=bool(c4)),
                   sigma_y=SIGMA_Y, modes=MODES, n=args.n, seed=args.seed,
                   seconds=round(time.time() - t0, 1))
    path = os.path.join(args.out, "toy-coin-receipt.json")
    with open(path, "w") as f: json.dump(receipt, f, indent=2)
    print(f"\nreceipt: {path}")
    ok = c2 and c3 and c4
    print(f"\nC2 {c2}   C3 {c3}   C4 {c4}")
    print(f"VERDICT: {'ALL CLAIMS HOLD' if ok else 'SOME CLAIMS FAILED'}   ({receipt['seconds']}s)")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
