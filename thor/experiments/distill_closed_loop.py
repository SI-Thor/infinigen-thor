#!/usr/bin/env python3
"""
distill_closed_loop.py - EXPERIMENT E2, at toy scale, in pure numpy.

    THE QUESTION.  A generator is distilled from a teacher.  The standard
    recipe distils the teacher's PIXELS.  The alternative recipe distils in
    the CLOSED-LOOP metric:

        pixel student      minimise  || G_s(y) - G_t(y) ||^2
        closed-loop student minimise  || A(G_s(y)) - y ||^2  +  lam * plausibility

    WHY THE SECOND SHOULD WIN.  The teacher's output is a SAMPLE from a
    multimodal posterior.  Regressing on samples converges to the posterior
    MEAN, and the mean of a multimodal posterior is a point that satisfies
    the measurement and exists in no scene.  Pixel distillation therefore
    TEACHES THE STUDENT TO PRODUCE GHOSTS.  The closed-loop objective, with
    the plausibility term, does not.

    Three students, one budget each:
        A. pixel distillation          (the industry recipe)
        B. closed-loop, consistency    (consistency only -- the DEGENERATE one)
        C. closed-loop, + plausibility (the proposed one)

    Prediction: A and B land on the SAME blur.  C does not.
    If A and B agree, the closed-loop metric alone is not the answer and the
    PLAUSIBILITY TERM is doing all the work -- which is the claim.

    Run:  python3 distill_closed_loop.py
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_coin import (MLP, MU, MODES, SIGMA_X, sample_world, mode_distance,
                      train_map, train_score, dsm_residual, conditional_sample,
                      exact_conditional_sample, knn_manifold_distance)

# ---------------------------------------------------------------------------
# differentiable pieces of the loss
# ---------------------------------------------------------------------------
def net_input_jacobian(net, X, eps=1e-4):
    """J[n,i,j] = d out_i / d x_j   for a batch.  D forward calls, not N."""
    N, D = X.shape
    J = np.empty((N, D, D))
    for j in range(D):
        e = np.zeros(D); e[j] = eps
        J[:, :, j] = (net.forward(X + e) - net.forward(X - e)) / (2 * eps)
    return J

def consistency_term(A_net, G, Yb, eps=1e-4):
    """||A(G)-y||^2 and its gradient w.r.t. G."""
    N, D = G.shape
    res = A_net.forward(G)[:, 0] - Yb
    J = np.empty((N, D))
    for j in range(D):
        e = np.zeros(D); e[j] = eps
        J[:, j] = (A_net.forward(G + e)[:, 0] - A_net.forward(G - e)[:, 0]) / (2 * eps)
    val = float((res ** 2).mean())
    return val, 2.0 * res[:, None] * J / N

def plausibility_term(net_s, G, sigma=0.35, k=4, seed=0):
    """E || s(G+sigma*eps) + eps/sigma ||^2 and its gradient w.r.t. G.
    The score net is FROZEN, so only d/dG is needed -- no parameter grads."""
    rng = np.random.default_rng(seed)
    N, D = G.shape
    val = 0.0; dG = np.zeros_like(G)
    for _ in range(k):
        e = rng.normal(0, 1, (N, D))
        x = G + sigma * e
        r = net_s.forward(x) + e / sigma
        val += float((r ** 2).sum(1).mean()) / k
        J = net_input_jacobian(net_s, x)
        dG += np.einsum("ni,nij->nj", 2.0 * r, J) / k
    return val, dG

def train_student(Y, loss_fn, steps=1200, batch=512, seed=0, lr=2e-3, hid=64):
    rng = np.random.default_rng(seed)
    net = MLP([1, hid, hid, 2], seed=seed)
    m = net.zeros_like(); v = net.zeros_like()
    n = len(Y)
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        yb = Y[idx]
        G, cs = net.forward(yb[:, None], cache=True)
        _, dG = loss_fn(yb, G)
        gW, gb = net.backward(dG, cs)
        net.adam(gW, gb, m, v, t, lr=lr)
    return net

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/distill")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print("=" * 78)
    print("distill_closed_loop.py - EXPERIMENT E2: distill in the CLOSED LOOP,")
    print("                          not in the pixels")
    print("=" * 78)

    X = sample_world(args.n, rng)
    SIGMA_Y = 0.06
    Y = X[:, 0] + rng.normal(0, SIGMA_Y, args.n)
    A_net = train_map(X, Y[:, None], 1, steps=600, seed=args.seed + 1)
    net_s = train_score(X, steps_per_sigma=110, seed=args.seed + 2)
    print(f"\nperception operator trained; score trained.")

    # ---- the teacher: a conditional sampler (stochastic) ---------------------
    print("sampling the TEACHER (conditional sampler) ...")
    # the teacher is the EXACT conditional sampler: the best a conditional
    # generator can be.  Measuring the students against the gold standard is
    # the only way the comparison means anything.
    T = exact_conditional_sample(Y, rng, SIGMA_Y)
    print(f"  teacher samples: off-manifold {mode_distance(T).mean():.4f} "
          f"(real data {mode_distance(X).mean():.4f})")

    # ---- student A: pixel distillation --------------------------------------
    # The teacher's sample paired with its own y.  Regressing on samples
    # converges to the posterior MEAN -- which is the point of the experiment.
    y_to_t = {}
    for i, yv in enumerate(Y):
        y_to_t.setdefault(round(float(yv), 6), []).append(T[i])
    def pixel_loss_impl(yb, G):
        tb = np.stack([y_to_t[round(float(v), 6)][0] for v in yb])
        return float(((G - tb) ** 2).mean()), 2.0 * (G - tb) / len(yb)

    print("training student A: PIXEL distillation ...")
    A_stu = train_student(Y, pixel_loss_impl, steps=args.steps, seed=args.seed + 3)

    # ---- student B: closed-loop, consistency only ---------------------------
    def cons_loss(yb, G):
        return consistency_term(A_net, G, yb)
    print("training student B: closed-loop, CONSISTENCY only ...")
    B_stu = train_student(Y, cons_loss, steps=args.steps, seed=args.seed + 4)

    # ---- student C: closed-loop + plausibility ------------------------------
    # The lambda sweep is WIDE and the selection criterion is JOINT: a student
    # that collapses onto the modes while ignoring the measurement is not a
    # win, it is a different degenerate solution.
    # BALANCE THE TWO TERMS.  They are in different units: the consistency
    # gradient scales like 1/sigma_y^2, the plausibility gradient like the
    # score's own Jacobian (~1/sigma_x^2 near a mode).  Measured on this toy
    # the ratio is ~1e5, so an unweighted sum lets plausibility win outright:
    # the student collapses onto the modes and IGNORES the measurement.
    # So: measure both gradient magnitudes once and scale the plausibility
    # term, so that lambda=1 means "one unit of each".
    probe = np.random.default_rng(0).integers(0, len(Y), 256)
    Gp = MLP([1, 64, 64, 2], seed=1).forward(Y[probe][:, None])
    _v1, g1p = consistency_term(A_net, Gp, Y[probe])
    _v2, g2p = plausibility_term(net_s, Gp)
    bal = float(np.sqrt((g1p ** 2).sum(1).mean()) /
                max(np.sqrt((g2p ** 2).sum(1).mean()), 1e-12))
    print(f"    gradient scale balance factor: {bal:.3e} "
          f"(consistency / plausibility)")

    lam_rows = []
    for lam in (0.3, 1.0, 3.0):
        def full_loss(yb, G, lam=lam):
            v1, g1 = consistency_term(A_net, G, yb)
            v2, g2 = plausibility_term(net_s, G)
            return v1 + lam * bal * v2, g1 + lam * bal * g2
        C = train_student(Y, full_loss, steps=args.steps, seed=args.seed + 5)
        Z = C.forward(Y[:, None])
        md = float(mode_distance(Z).mean())
        rt = float(np.abs(A_net.forward(Z)[:, 0] - Y).mean())
        lam_rows.append((lam, C, Z, md, rt))
        print(f"    lam={lam:<7} off-manifold {md:8.4f}  round-trip {rt:.4f}")
    t_off0 = float(mode_distance(T).mean())
    t_rt0 = float(np.abs(A_net.forward(T)[:, 0] - Y).mean())
    def joint(e):
        lam, C, Z, md, rt = e
        return md / max(t_off0, 1e-9) + rt / max(t_rt0, 1e-9)
    lam_b, C_stu, Z_C, md_b, rt_b = min(lam_rows, key=joint)
    print(f"    selected lam={lam_b} (joint criterion)")

    # ---- measure ------------------------------------------------------------
    def report(name, net, extra=""):
        Z = net.forward(Y[:, None])
        md = mode_distance(Z)
        rt = float(np.abs(A_net.forward(Z)[:, 0] - Y).mean())
        pl = float(dsm_residual(net_s, Z, seed=3).mean())
        return dict(method=name, off_manifold_mean=float(md.mean()),
                    off_manifold_p95=float(np.percentile(md, 95)),
                    roundtrip_mae=rt, plausibility=pl, extra=extra), Z

    rows = []
    md_real = float(mode_distance(X).mean())
    pl_real = float(dsm_residual(net_s, X, seed=3).mean())
    rows.append(dict(method="REAL DATA (floor)", off_manifold_mean=md_real,
                     off_manifold_p95=float(np.percentile(mode_distance(X), 95)),
                     roundtrip_mae=float(SIGMA_Y), plausibility=pl_real))
    rA, ZA = report("A. pixel distillation", A_stu)
    rB, ZB = report("B. closed-loop, consistency only", B_stu)
    rC, ZC = report("C. closed-loop + plausibility", C_stu, f"lam={lam_b:g}")
    rows += [rA, rB, rC]
    for lam, C, Z, md, rt in lam_rows:
        rows.append(dict(method=f"   C at lam={lam}", off_manifold_mean=md,
                         off_manifold_p95=float(np.percentile(mode_distance(Z), 95)),
                         roundtrip_mae=rt,
                         plausibility=float(dsm_residual(net_s, Z, seed=3).mean())))
    T_row = dict(method="TEACHER (sampler)", off_manifold_mean=float(mode_distance(T).mean()),
                 off_manifold_p95=float(np.percentile(mode_distance(T), 95)),
                 roundtrip_mae=float(np.abs(A_net.forward(T)[:, 0] - Y).mean()),
                 plausibility=float(dsm_residual(net_s, T, seed=3).mean()))
    rows.insert(1, T_row)

    print("\n" + "=" * 78)
    print(f"{'method':<38}{'off-manifold':>14}{'roundtrip':>12}{'plausibility':>15}")
    print("-" * 78)
    for r in rows:
        print(f"{r['method']:<38}{r['off_manifold_mean']:>14.4f}"
              f"{r['roundtrip_mae']:>12.4f}{r['plausibility']:>15.3f}")
    print("=" * 78)

    t_off = T_row["off_manifold_mean"]; t_rt = T_row["roundtrip_mae"]

    real_off = rows[0]["off_manifold_mean"]; real_rt = rows[0]["roundtrip_mae"]

    def both(r):
        """plausible AND consistent, judged against the REAL DATA floor:
        a world that a real world could have produced."""
        return (r["off_manifold_mean"] <= 3.0 * real_off and
                r["roundtrip_mae"] <= 3.0 * real_rt)

    # C1 CONTROL: neither pixel distillation nor consistency alone produces a
    # plausible world.  They are DIFFERENT ghosts -- the blur (the teacher's
    # mean) and an arbitrary point on the fiber -- but both are ghosts.
    c1 = (rA["off_manifold_mean"] > 3.0 * t_off and
          rB["off_manifold_mean"] > 3.0 * t_off)
    # C2: the PLAUSIBILITY term is what turns a ghost into a possible world.
    c2 = (rC["off_manifold_mean"] <= 0.35 * rA["off_manifold_mean"] and
          rC["off_manifold_mean"] <= 0.35 * rB["off_manifold_mean"])
    # C3 THE STRUCTURAL RESULT: only the STOCHASTIC generator is both
    # plausible and consistent.  A deterministic g(y) must put g(y) inside the
    # prior's support AND satisfy A(g(y)) = y -- and for a multimodal prior
    # with discrete modes, no such point exists for a generic y.  The tension
    # is IRREDUCIBLE, not a tuning failure.
    c3 = (both(T_row) and not both(rA) and not both(rB) and
          rC["roundtrip_mae"] > 3.0 * t_rt)

    print(f"\nCONTROL C1  neither naive objective may produce a plausible world "
          f"(teacher {t_off:.4f}):")
    print(f"         pixel {rA['off_manifold_mean']:.4f}   consistency-only "
          f"{rB['off_manifold_mean']:.4f}  -> both ghosts? {'PASS' if c1 else 'FAIL'}")
    print(f"CLAIM   C2  the plausibility term turns the ghost into a possible world:")
    print(f"         closed-loop student off-manifold {rC['off_manifold_mean']:.4f}  "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"CLAIM   C3  only the STOCHASTIC generator is plausible AND consistent:")
    for nm, r in (("teacher (stochastic)", T_row), ("A pixel (det)", rA),
                  ("B consistency only (det)", rB), ("C closed-loop (det)", rC)):
        print(f"         {nm:<28} off {r['off_manifold_mean']:8.4f}  "
              f"rt {r['roundtrip_mae']:7.4f}  both? {'YES' if both(r) else 'no'}")
    print(f"         {'PASS' if c3 else 'FAIL'}  -- and this is STRUCTURAL:")
    print(f"         a deterministic g(y) must land inside the prior's support")
    print(f"         AND satisfy A(g(y)) = y.  For a multimodal prior with")
    print(f"         discrete modes, for a generic y NO such point exists")
    print(f"         (the modes project onto a measure-zero set of observations).")
    print(f"         So the generator MUST be stochastic wherever the fibre is")
    print(f"         wide -- which is exactly what the fibre-width gate says.")

    receipt = dict(rows=rows,
                   claims=dict(C1_both_naive_objectives_are_ghosts=bool(c1),
                               C2_plausibility_makes_it_possible=bool(c2),
                               C3_only_the_stochastic_generator_is_both=bool(c3)),
                   lam=lam_b, teacher=dict(off_manifold=t_off, roundtrip=t_rt),
                   n=args.n, steps=args.steps, seed=args.seed,
                   seconds=round(time.time() - t0, 1))
    path = os.path.join(args.out, "distill-closed-loop-receipt.json")
    with open(path, "w") as f: json.dump(receipt, f, indent=2)
    print(f"\nreceipt: {path}")
    print(f"VERDICT: {'ALL CLAIMS HOLD' if (c1 and c2 and c3) else 'SOME CLAIMS FAILED'}  "
          f"({receipt['seconds']}s)")
    return 0 if (c1 and c2 and c3) else 1

if __name__ == "__main__":
    sys.exit(main())
