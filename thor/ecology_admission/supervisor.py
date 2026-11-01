"""Admission supervisor — the ecology-thor policy applied to infinigen jobs.

External by design: this process never touches infinigen internals. It
observes job events (log lines / stage completions / process liveness),
keeps the WHY ledger, and decides:

  * who gets admitted next (attention auction under GPU-second rails),
  * who has stalled (freshness slide -> UNKNOWN at the horizon),
  * what happens then (SIGUSR1 -> --on_interrupt checkpoint -> park),
  * who may revive (explicit, on parole).

The memory axis is not ours: since builder-ide's ``thor-admit`` landed,
the box has a memory admission daemon (kernel-held leases, the 250 ms
watchdog, learned peaks). We compose with it once per admission — the
auction decides a job is worth its GPU-seconds, then the supervisor
asks thor-admit whether the box has room for its peak
(:mod:`.lease`). A lease refused is not a scheduling failure: the job
stays queued and its purpose stays fresh.

The dry-run mode (`enforcing=False`, the default in the simulator) makes
it a pure logger of *decisions* — safe to run against a live batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .lease import LEASE_PRIORITY, LeaseClient, RecordingLeaseClient
from .ledger import WhyLedger, WhyState

IMPORTANCE = {"kernel": 3, "world": 2, "scenery": 1, "maintenance": 0}


@dataclass
class Rails:
    """The box's real currencies (builder-ide-compatible units)."""

    gpu_seconds: float = 0.0
    joules: float = 0.0
    cents: float = 0.0

    def afford(self, cost: "Rails") -> bool:
        return (
            self.gpu_seconds >= cost.gpu_seconds
            and self.joules >= cost.joules
            and self.cents >= cost.cents
        )

    def spend(self, cost: "Rails") -> None:
        self.gpu_seconds -= cost.gpu_seconds
        self.joules -= cost.joules
        self.cents -= cost.cents


@dataclass
class Job:
    """One infinigen scene job (a purpose node in the ledger)."""

    ledger_id: int
    name: str
    region: str
    stage: str = "coarse"
    gpu_seconds_cost: float = 1.0
    joules_cost: float = 25.0
    urgency: float = 0.5
    opportunity: float = 0.8
    novelty: float = 0.7
    expected_value: float = 0.6
    importance: str = "world"
    pid: Optional[int] = None  # None until launched (dry-run keeps None)
    checkpointed: bool = False
    lease_id: Optional[int] = None  # thor-admit lease while running


@dataclass
class Decision:
    tick: int
    kind: str  # admit | harvest | checkpoint | revive | deadline | veto
    job: str
    detail: str


@dataclass
class AdmissionReport:
    tick: int
    admitted: List[str] = field(default_factory=list)
    vetoed: int = 0
    skipped_infeasible: int = 0
    spent: Rails = field(default_factory=Rails)


class Supervisor:
    """Heartbeat + auction + harvest policy over a :class:`WhyLedger`."""

    def __init__(
        self,
        horizon: int = 900,
        deadline_grace: int = 300,
        cascade_budget: int = 64,
        rails_refill_per_period: float = 4.0,
        max_admitted: int = 6,
        enforcing: bool = False,
        lease_client: Optional[LeaseClient] = None,
    ):
        self.ledger = WhyLedger(
            horizon=horizon, deadline_grace=deadline_grace, cascade_budget=cascade_budget
        )
        self.rails = Rails()
        self.rails_refill = Rails(
            gpu_seconds=rails_refill_per_period, joules=rails_refill_per_period * 25.0
        )
        self.max_admitted = max_admitted
        self.enforcing = enforcing
        # The memory axis: thor-admit's, by lease. The default records what
        # we would have leased (shadow mode); enforcing mode passes the real
        # client and a refused lease leaves the job queued.
        self.leases: LeaseClient = lease_client or RecordingLeaseClient()
        self.jobs: Dict[int, Job] = {}  # ledger_id -> Job
        self.running: List[int] = []  # ledger ids currently admitted
        self.decisions: List[Decision] = []

    # ------------------------------------------------------------- worlds

    def add_region(self, until: Optional[int], now: int) -> int:
        return self.ledger.spawn(parent=None, now=now, until=until, mass=10.0)

    def add_job(
        self,
        region_ledger_id: int,
        name: str,
        now: int,
        stage: str = "render",
        gpu_seconds_cost: float = 1.0,
        urgency: float = 0.5,
        novelty: float = 0.7,
        importance: str = "world",
    ) -> int:
        jid = self.ledger.spawn(parent=region_ledger_id, now=now, mass=1.0)
        self.jobs[jid] = Job(
            ledger_id=jid,
            name=name,
            region=str(region_ledger_id),
            stage=stage,
            gpu_seconds_cost=gpu_seconds_cost,
            urgency=urgency,
            novelty=novelty,
            importance=importance,
        )
        return jid

    # -------------------------------------------------------- attestation

    def job_progress(self, name: str, now: int) -> bool:
        """A job emitted a log line / finished a stage: service its chain."""
        for jid, job in self.jobs.items():
            if job.name == name:
                return self.ledger.run_under(jid, now) > 0
        return False

    def job_done(self, name: str, now: int) -> bool:
        """Job succeeded: withdraw its bid, free its slot, release the lease."""
        for jid, job in self.jobs.items():
            if job.name == name:
                if jid in self.running:
                    self.running.remove(jid)
                if job.lease_id is not None:
                    self.leases.release(job.lease_id)
                    job.lease_id = None
                return self.ledger.run_under(jid, now) > 0
        return False

    # -------------------------------------------------------------- policy

    def period(self, now: int) -> AdmissionReport:
        """One scheduling period: refill, reap, deadlines, cascade, auction."""
        report = AdmissionReport(tick=now)

        # Rails income for this period.
        self.rails.gpu_seconds += self.rails_refill.gpu_seconds
        self.rails.joules += self.rails_refill.joules

        # The economic harvest first: stalls become UNKNOWN (questionable).
        for lapsed in self.ledger.reap(now):
            job = self.jobs.get(lapsed)
            if job is None:
                continue  # a region root lapsed; handled by deadline_check next
            self.decisions.append(
                Decision(now, "harvest", job.name, "lapsed to UNKNOWN (no progress past horizon)")
            )
            self._checkpoint_and_park(job, now)
            if lapsed in self.running:
                self.running.remove(lapsed)

        # Root deadlines: regions that missed `until` die, cascading doubt.
        for dead in self.ledger.deadline_check(now):
            self.decisions.append(
                Decision(now, "deadline", f"region#{dead}", "root deadline missed; cascading")
            )
            for jid in list(self.running):
                if self.jobs[jid].region == str(dead):
                    self.running.remove(jid)

        # Continue any rate-limited cascade.
        self.ledger.drain_cascade(now)

        # Free capacity?
        free = self.max_admitted - len(self.running)

        # The auction: candidates are ALIVE jobs not already running.
        candidates: List[Tuple[int, float, float, int]] = []  # (class, score, cost, jid)
        for jid, job in self.jobs.items():
            if self.ledger.state(jid) is not WhyState.ALIVE or jid in self.running:
                continue
            j = self.ledger.justification(jid, now)
            score = job.urgency * j * job.opportunity * job.novelty * job.expected_value
            if score <= 0.0:
                report.vetoed += 1  # THE VETO LAW: zero-attention never transacts
                continue
            candidates.append((IMPORTANCE[job.importance], score, job.gpu_seconds_cost, jid))

        # Deterministic strict order: class desc, score desc, id asc.
        candidates.sort(key=lambda c: (-c[0], -c[1], c[3]))

        admitted = 0
        for _klass, score, cost, jid in candidates:
            if admitted >= free:
                break
            job = self.jobs[jid]
            rail_cost = Rails(gpu_seconds=cost, joules=job.joules_cost)
            if not self.rails.afford(rail_cost):
                report.skipped_infeasible += 1
                continue
            # The memory axis, once per admission: the auction has decided
            # the job is worth its GPU-seconds; now does the box have room
            # for its peak? A refusal is not a scheduling failure — the job
            # stays queued, its purpose stays fresh.
            grant = self.leases.acquire(
                label=job.name, peak="auto", priority=LEASE_PRIORITY[job.importance]
            )
            if not grant.ok:
                self.decisions.append(
                    Decision(now, "lease", job.name, f"thor-admit refused: {grant.reason}")
                )
                report.skipped_infeasible += 1
                continue
            job.lease_id = getattr(grant, "lease_id", None)
            self.rails.spend(rail_cost)
            self.running.append(jid)
            report.admitted.append(job.name)
            report.spent.spend(rail_cost)
            admitted += 1
            self.decisions.append(
                Decision(
                    now,
                    "admit",
                    job.name,
                    f"score={score:.3f} class={job.importance} cost={cost:.1f}gpu-s"
                    + (
                        f" lease={job.lease_id}"
                        if job.lease_id is not None
                        else ""
                    ),
                )
            )
        return report

    # ------------------------------------------------------------ actions

    def _checkpoint_and_park(self, job: Job, now: int) -> None:
        """SIGUSR1 -> --on_interrupt checkpoint -> park (never delete)."""
        if self.enforcing and job.pid is not None:  # pragma: no cover - real mode
            import os
            import signal

            try:
                os.kill(job.pid, signal.SIGUSR1)  # infinigen checkpoints & exits
            except ProcessLookupError:
                pass
        if job.lease_id is not None:
            # The lease goes back while the job is parked: a checkpointed job
            # holds no memory it isn't using. Revive asks for a new lease.
            self.leases.release(job.lease_id)
            job.lease_id = None
        job.checkpointed = True
        self.decisions.append(
            Decision(
                now,
                "checkpoint",
                job.name,
                "SIGUSR1 -> --on_interrupt checkpoint; parked (never deleted)",
            )
        )

    def revive(self, name: str, now: int) -> bool:
        """A human or policy decides a harvested job gets another chance."""
        for jid, job in self.jobs.items():
            if job.name == name:
                ok = self.ledger.revive(jid, now)
                if ok:
                    self.decisions.append(
                        Decision(now, "revive", name, "revived on parole (half-strength bids)")
                    )
                return ok
        return False
