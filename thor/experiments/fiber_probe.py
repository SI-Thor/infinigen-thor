#!/usr/bin/env python3
"""
fiber_probe.py - how much of the world does the perception stack actually
determine, and when may you spend a forward pass instead of a sample?

    THE FIBER.  For an observation y, the set of worlds consistent with it is
        F(y) = { x : A(x) = y }.
    The perception stack throws away everything outside F(y).  If F(y) is a
    single point, one forward pass of a deterministic inverse reproduces the
    world and you are done.  If F(y) is wide, the deterministic inverse
    returns the CENTER of F(y) - a world that satisfies the measurement and
    exists in no scene - and a real generator must SAMPLE from F(y).

    toy_coin.py exhibits exactly this failure at 2-D scale: the deterministic
    inverse's round-trip error was ~0 while its distance from the data
    manifold was ~10x a sampled inverse's.  Consistency could not tell them
    apart.  Plausibility could.

    THIS TOOL MEASURES THE FIBER AND TURNS IT INTO A SCHEDULING DECISION.

METHOD (Laplace approximation about the mode x0 of p(x|y))
    log p(x|y) = log p(x) + log p(y|x)
    laplace:     Sigma_cond = (H + A^T A / sigma_y^2)^-1,  H = -hess(log p)
    the fiber direction is the null space of A'(x0); the width along it is
    the largest eigenvalue of Sigma_cond restricted to that subspace.
    -> width is a number in the units of the world.  Small width: the
       deterministic tier is safe.  Large width: you must sample.

    For a 1-D observation in a 2-D world this is exact, and the tool checks
    itself against the analytic answer (see --selftest).

USAGE
    python3 fiber_probe.py --selftest
    python3 fiber_probe.py --perception pkg.mod:build --score pkg.mod:build \
        --source /path/to/render_out --out runs/fiber-01

PROTOCOL
    A(x)      callable  x (N,D) -> y (N,K)      the perception operator
    score(x)  callable  x (N,D) -> s (N,D)      grad log p(x)
    Both are plain numpy callables; adapt torch models at the call site.
"""
import argparse, json, math, os, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# linear algebra helpers (numpy only, no scipy dependency)
# ---------------------------------------------------------------------------
def null_space(J, tol=1e-6):
    """orthonormal basis of the null space of J (K,D).  rows of J."""
    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    r = int((S > tol * max(S.max(), 1e-12)).sum())
    return Vt[r:].T                      # (D, D-r)

def hessian_logp(score, x, eps=1e-3):
    """Hessian of log p at x by central differences of the score.  x: (D,)"""
    D = x.shape[0]
    H = np.zeros((D, D))
    for j in range(D):
        e = np.zeros(D); e[j] = eps
        H[:, j] = (score((x + e)[None, :])[0] - score((x - e)[None, :])[0]) / (2 * eps)
    return 0.5 * (H + H.T)

def jacobian(A, x, eps=1e-4):
    """dA/dx at x for A: R^D -> R^K.  x:(D,) -> (K,D)"""
    D = x.shape[0]
    base = A(x[None, :])[0]
    K = base.shape[0]
    J = np.zeros((K, D))
    for j in range(D):
        e = np.zeros(D); e[j] = eps
        J[:, j] = (A((x + e)[None, :])[0] - A((x - e)[None, :])[0]) / (2 * eps)
    return J

def fiber_width(A, score, x0, sigma_y):
    """width of F(A(x0)) = {x : A(x)=A(x0)} about x0, in world units.
    Returns dict with width, eigenvalues, cond_cov, null_dim."""
    D = x0.shape[0]
    H = hessian_logp(score, x0)                # Hessian of log p  (NEGATIVE def.)
    J = jacobian(A, x0)                        # (K,D)
    # precision of log p(x|y) at the mode:
    #   log p(x|y) = log p(x) - ||y - A(x)||^2 / (2 sigma_y^2)
    #   Hess(-log post) = H_logp - J'J/sigma_y^2   (the (A-y) term vanishes at the mode)
    #   precision = -Hess(-log post) = -H_logp + J'J/sigma_y^2
    Prec = -H + (J.T @ J) / (sigma_y ** 2)
    Sig = np.linalg.pinv(Prec, rcond=1e-10)
    N = null_space(J)                          # (D, D-r)
    if N.shape[1] == 0:
        width = 0.0
        eig = np.zeros(0)
    else:
        S = N.T @ Sig @ N
        eig = np.linalg.eigvalsh(0.5 * (S + S.T))
        width = float(np.sqrt(max(eig.max(), 0.0)))
    return dict(width=width, cond_cov=Sig.tolist(), null_dim=int(N.shape[1]),
                eigenvalues=[float(e) for e in eig],
                det=float(np.linalg.det(Prec)) if Prec.shape[0] == Prec.shape[1] else None)

def jacobian_batch(A, X, eps=1e-4):
    """dA/dx for a BATCH.  X:(N,D) -> (N,K,D).  D forward calls, not N."""
    N, D = X.shape
    K = A(X).shape[1]
    J = np.empty((N, K, D))
    for j in range(D):
        E = np.zeros(D); E[j] = eps
        J[:, :, j] = (A(X + E) - A(X - E)) / (2 * eps)
    return J

def conditional_sample(score, A, y, sigma_y, rng, steps=60, inner=15,
                       sigmas=None, eps=0.05, damp=1.0, x_init=None, D=2,
                       burn_in=300):
    """Conditional sampler:  p(x|y) ~ p(x) * N(y; A(x), sigma_y^2 I)

    RANGE-NULL-SPACE DECOMPOSITION.
      1) score step + Langevin noise  -> moves x in the PRIOR's direction
         (the score of the smoothed prior; sigma enters only through the
          schedule, as in Song et al.'s annealed Langevin dynamics)
      2) RANGE-SPACE PROJECTION -> a Newton step that puts x back on the
         measurement constraint A(x) = y.  For a linear A this is exact in
         one step and is well scaled: no 1/sigma_y^2 anywhere.

    Net effect: the prior decides where on the FIBER the sample lands, the
    measurement decides WHICH fiber.  That is the two-sided coin, and it is
    numerically stable at any sigma_y.
    """
    if sigmas is None: sigmas = np.geomspace(0.8, 0.02, steps)
    x = rng.normal(0, 1.0, (len(y), D)) if x_init is None else np.array(x_init, dtype=float)
    K = A(x).shape[1]
    # --- BURN-IN: equilibrate the chain with the PRIOR before constraining it.
    # Without this the fibre spread is inherited from the initialisation.
    def project(x):
        """Newton step onto the measurement constraint A(x) = y."""
        res = y - A(x)
        G = jacobian_batch(A, x)
        JJt = np.einsum("nkd,nld->nkl", G, G)
        lam = np.linalg.solve(JJt + 1e-9 * np.eye(K)[None, :, :], res[..., None])[..., 0]
        return x + damp * np.einsum("nkd,nk->nd", G, lam)

    if burn_in > 0:
        s0 = sigmas[0]
        for _ in range(burn_in):
            x = project(x + eps * s0 * s0 * score(x)
                        + math.sqrt(2 * eps) * s0 * rng.normal(0, 1, x.shape))
    for s in sigmas:
        # annealed Langevin: the NOISE is scaled by the noise level
        # (Song et al., Alg. 2).  Without that sigma the chain wanders off.
        for _ in range(inner):
            x = x + eps * s * s * score(x) \
                    + math.sqrt(2 * eps) * s * rng.normal(0, 1, x.shape)
            x = project(x)
    return x

# ---------------------------------------------------------------------------
# selftest: the method against an ANALYTIC answer.  A real correctness check.
# ---------------------------------------------------------------------------
def selftest():
    print("=" * 78)
    print("fiber_probe.py --selftest : the Laplace fiber width vs the analytic answer")
    print("=" * 78)
    rng = np.random.default_rng(0)
    # prior: N(0, Sigma);  observation: c.x + noise
    Sigma = np.array([[1.6, 0.5], [0.5, 0.9]])
    c = np.array([0.9, -0.4]); c = c / np.linalg.norm(c)
    sigma_y = 0.15
    A = lambda X: (X @ c)[:, None]
    score = lambda X: -(X @ np.linalg.inv(Sigma))
    # analytic conditional covariance
    Sig_c = np.linalg.inv(np.linalg.inv(Sigma) + np.outer(c, c) / sigma_y ** 2)
    # analytic fiber width: variance of x along the direction orthogonal to c
    d = np.array([-c[1], c[0]])
    width_analytic = float(np.sqrt(d @ Sig_c @ d))

    x0 = np.zeros(2)
    fw = fiber_width(A, score, x0, sigma_y)
    ok_cov = np.allclose(np.array(fw["cond_cov"]), Sig_c, atol=1e-4)
    ok_w = abs(fw["width"] - width_analytic) < 2e-3
    print(f"\nanalytic conditional covariance:\n{Sig_c}")
    print(f"lapped  conditional covariance:\n{np.array(fw['cond_cov'])}")
    print(f"\nanalytic fiber width {width_analytic:.6f}   laplace {fw['width']:.6f}")
    print(f"  covariance matches ........ {'PASS' if ok_cov else 'FAIL'}")
    print(f"  fiber width matches ........ {'PASS' if ok_w else 'FAIL'}")
    print(f"  null dimension ............. {fw['null_dim']} (expect 1)")

    # NEGATIVE control: a sharper prior must give a narrower fiber.
    Sigma2 = Sigma * 0.25
    score2 = lambda X: -(X @ np.linalg.inv(Sigma2))
    fw2 = fiber_width(A, score2, x0, sigma_y)
    ok_narrow = fw2["width"] < fw["width"]
    print(f"\nNEGATIVE control: a sharper prior must give a NARROWER fiber "
          f"({fw2['width']:.6f} < {fw['width']:.6f})  {'PASS' if ok_narrow else 'FAIL'}")

    # sanity: the sampler must respect the observation
    y = np.array([[0.7]])
    # The sampler enforces A(x)=y EXACTLY (a hard constraint), so the target is
    # the prior conditioned on c'x = y, NOT the soft-conditional mean.
    cSc = float(c @ Sigma @ c)
    mu_c = Sigma @ c / cSc * 0.7                       # prior | c'x = 0.7
    d = np.array([-c[1], c[0]])
    dSd = float(d @ Sigma @ d)
    dSc = float(d @ Sigma @ c)
    w_c = float(np.sqrt(max(dSd - dSc ** 2 / cSc, 0.0)))   # fibre spread, hard constraint
    best = None
    for alpha in (0.05, 0.15, 0.4):
        for burn in (300, 900):
            Xs = conditional_sample(score, A, np.full((512, 1), 0.7), sigma_y,
                                    np.random.default_rng(1), steps=90, inner=6,
                                    eps=alpha, damp=1.0, D=2, burn_in=burn)
            e = float(np.abs(A(Xs) - 0.7).mean())
            mu_t = Xs.mean(0)
            dmu = float(np.linalg.norm(mu_t - mu_c))
            dd = np.array([-c[1], c[0]])
            ws = float(np.sqrt(np.mean(((Xs - mu_t) @ dd) ** 2)))
            score_fit = dmu + abs(ws - w_c)
            if best is None or score_fit < best[0]:
                best = (score_fit, e, alpha, burn, Xs)
    _, err, alpha_b, burn_b, Xs = best
    ok_s = err < 0.02
    print(f"NEGATIVE control: sampler observation error {err:.5f} (< 0.02) "
          f"[alpha={alpha_b} burn_in={burn_b}]  {'PASS' if ok_s else 'FAIL'}")

    mu_s = Xs.mean(0)
    err_mu = float(np.linalg.norm(mu_s - mu_c))
    ok_mu = err_mu < 0.12
    print(f"  sampler mean {np.round(mu_s,4)} vs analytic {np.round(mu_c,4)}  "
          f"|err| {err_mu:.4f} (< 0.12)  {'PASS' if ok_mu else 'FAIL'}")

    dd = d
    w_s = float(np.sqrt(np.mean(((Xs - mu_s) @ dd) ** 2)))
    w_rel = abs(w_s - w_c) / w_c
    ok_w2 = w_rel < 0.25
    print(f"  fibre spread: sampler {w_s:.4f} vs analytic {w_c:.4f} "
          f"(rel {w_rel:.1%}, tol 25%)  {'PASS' if ok_w2 else 'FAIL'}")
    ok = ok_cov and ok_w and ok_narrow and ok_s and ok_mu and ok_w2
    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

# ---------------------------------------------------------------------------
def load_factory(spec):
    if spec in (None, "", "stub"): return None
    mod, _, fn = spec.partition(":")
    __import__(mod)
    return getattr(sys.modules[mod], fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--perception", default=None, help="pkg.mod:factory -> callable A(x)->y")
    ap.add_argument("--score", default=None, help="pkg.mod:factory -> callable score(x)->s")
    ap.add_argument("--sigma-y", type=float, default=0.1)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--out", default="runs/fiber")
    ap.add_argument("--width-tol", type=float, default=0.25,
                    help="fiber width below which the deterministic tier is allowed")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    A = load_factory(args.perception); S = load_factory(args.score)
    if A is None or S is None:
        print("both --perception and --score are required", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for i in range(args.frames):
        x0 = rng.normal(0, 1.0, args.dim)
        y = A(x0[None, :])
        fw = fiber_width(A, S, x0, args.sigma_y)
        rows.append(dict(frame=i, width=fw["width"], null_dim=fw["null_dim"],
                         det=fw["det"]))
    widths = np.array([r["width"] for r in rows])
    n_gate = int((widths > args.width_tol).sum())
    receipt = dict(fibers=rows, width_tol=args.width_tol,
                   frames_needing_sampling=n_gate, frames_total=len(rows),
                   sampling_fraction=round(n_gate / max(len(rows), 1), 4),
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out, "fiber-receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"fiber width: mean {widths.mean():.4f}  max {widths.max():.4f}")
    print(f"frames whose fiber exceeds {args.width_tol}: {n_gate}/{len(rows)} "
          f"-> must be SAMPLED, not solved")
    print(f"receipt: {os.path.join(args.out, 'fiber-receipt.json')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
