"""Dry-run simulator: an overnight infinigen batch under the supervisor.

Fake clock, no infinigen import, no GPU. Models the shape the spike
measured (coarse CPU-serial, renders GPU-bound minutes-long, occasional
EXACT-Boolean-style silent stalls) and prints the decisions the
supervisor *would* make.

Run:  python -m thor.ecology_admission.sim --ticks 3600 --stall-rate 0.01
"""

from __future__ import annotations

import argparse
import random


def main() -> int:
    ap = argparse.ArgumentParser(description="ecology-admission dry-run simulator")
    ap.add_argument("--ticks", type=int, default=3600, help="simulated ticks (1 tick = 1 s)")
    ap.add_argument("--regions", type=int, default=3)
    ap.add_argument("--jobs-per-region", type=int, default=12)
    ap.add_argument("--stall-rate", type=float, default=0.02, help="P(stall) per admitted job per 100s")
    ap.add_argument("--horizon", type=int, default=900, help="attestation horizon (ticks)")
    ap.add_argument("--period", type=int, default=100, help="scheduling period (ticks)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from .supervisor import Supervisor

    rng = random.Random(args.seed)
    sup = Supervisor(horizon=args.horizon, rails_refill_per_period=6.0, max_admitted=6)
    regions = [sup.add_region(until=args.ticks, now=0) for _ in range(args.regions)]

    # Every job registers with the supervisor up front; the auction decides
    # order, not a queue. (Stalled set: the silent-hang class.)
    for r, rid in enumerate(regions):
        for i in range(args.jobs_per_region):
            sup.add_job(rid, f"r{r}-scene{i:02d}", now=0, gpu_seconds_cost=1.0, urgency=0.5 + 0.5 * rng.random())
    stalled: set[str] = set()
    done: set[str] = set()
    # admitted_at: name -> tick admitted (renders take ~120 ticks).
    admitted_at: dict[str, int] = {}

    total_admitted = 0
    for now in range(1, args.ticks + 1):
        # Admitted jobs make progress (or stall).
        for name in list(admitted_at):
            if name in stalled:
                continue
            if rng.random() < 0.5:
                sup.job_progress(name, now)  # log line -> heartbeat
            if now - admitted_at[name] > 120 and rng.random() < 0.3:
                # Render finished.
                sup.job_done(name, now)
                done.add(name)
                del admitted_at[name]
            elif rng.random() < args.stall_rate / 100.0:
                stalled.add(name)  # silent hang: no more heartbeats ever

        if now % args.period == 0:
            report = sup.period(now)
            total_admitted += len(report.admitted)
            for name in report.admitted:
                admitted_at[name] = now
                stalled.discard(name)  # fresh admission resets stall bookkeeping
            # The night-operator policy: checkpointed jobs occasionally get
            # another chance (revive on parole, explicit — never automatic).
            for jid, job in list(sup.jobs.items()):
                if sup.ledger.state(jid) is not None and str(sup.ledger.state(jid)).endswith("UNKNOWN"):
                    if rng.random() < 0.3:
                        sup.revive(job.name, now)

        # Region goals spawn more scenes as capacity opens.
        if now % 500 == 0:
            r = rng.randrange(len(regions))
            sup.add_job(regions[r], f"r{r}-scene{now:04d}", now=now)

    print("=== ecology-admission dry run ===")
    print(f"ticks simulated      : {args.ticks}")
    print(f"jobs admitted        : {total_admitted}")
    print(f"jobs completed       : {len(done)}")
    total_harvested = sum(1 for d in sup.decisions if d.kind == "harvest")
    print(f"silent stalls handled: {total_harvested} (each: UNKNOWN -> checkpoint -> park)")
    print(f"ledger end state     : {sup.ledger.counts()}")
    print(f"cascade still pending: {sup.ledger.cascade_pending}")
    print("\n--- last 12 decisions ---")
    for d in sup.decisions[-12:]:
        print(f"[{d.tick:>5}] {d.kind:<10} {d.job:<14} {d.detail}")
    print("\n(full log: sup.decisions — this is the artifact shadow mode writes to disk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
