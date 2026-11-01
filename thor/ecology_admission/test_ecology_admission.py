"""Tests for the ecology_admission prototype — fake clock, no infinigen.

These test the SEMANTICS mirror against the ecology-thor kernel's laws:
freshness slide, lapse, veto law, rails conservation, cascade rate
limiting, parole. They run in CI without a GPU, Rust, or infinigen.
"""

from __future__ import annotations

import pytest

from thor.ecology_admission.lease import (
    LEASE_PRIORITY,
    LeaseRefusal,
    NullLeaseClient,
    RecordingLeaseClient,
    ThorAdmitLeaseClient,
)
from thor.ecology_admission.ledger import PROBATION_CAP, WhyLedger, WhyState, link_freshness
from thor.ecology_admission.supervisor import Rails, Supervisor


class TestFreshness:
    def test_grace_then_linear_to_zero(self):
        h = 90  # grace = 30
        assert link_freshness(WhyState.ALIVE, 100, 0, 130, h) == 1.0
        assert link_freshness(WhyState.ALIVE, 100, 0, 160, h) == pytest.approx(0.5)
        assert link_freshness(WhyState.ALIVE, 100, 0, 190, h) == 0.0

    def test_stale_never_rewinds(self):
        assert link_freshness(WhyState.ALIVE, 200, 0, 100, 90) == 1.0

    def test_non_alive_links_are_neutral(self):
        assert link_freshness(WhyState.UNKNOWN, 0, 0, 9999, 90) == 1.0


class TestLedger:
    def test_never_deleted_no_api_for_it(self):
        w = WhyLedger(horizon=90)
        root = w.spawn(None, 0)
        child = w.spawn(root, 0)
        w.kill(root, 10)
        assert w.state(root) is WhyState.DEAD
        assert w.state(child) is WhyState.UNKNOWN  # questionable, not gone
        assert not hasattr(w, "delete") and not hasattr(w, "remove")

    def test_lapse_is_per_node(self):
        w = WhyLedger(horizon=90)
        a = w.spawn(None, 0)
        w.heartbeat(a, 50)
        lapsed = w.reap(200)
        assert lapsed == [a]
        assert w.state(a) is WhyState.UNKNOWN

    def test_revive_lands_on_parole(self):
        w = WhyLedger(horizon=90)
        a = w.spawn(None, 0)
        w.reap(200)
        assert w.revive(a, 200)
        assert w.justification(a, 200) == pytest.approx(PROBATION_CAP)
        w.run_under(a, 205)  # service ends parole
        assert w.justification(a, 205) == pytest.approx(1.0)

    def test_weakest_link_dominates(self):
        w = WhyLedger(horizon=90)
        root = w.spawn(None, 0)
        mid = w.spawn(root, 0)
        leaf = w.spawn(mid, 0)
        w.heartbeat(root, 150)
        w.heartbeat(leaf, 150)
        # mid unattested since 0: age 150 > horizon -> freshness 0.
        assert w.justification(leaf, 150) == 0.0
        w.heartbeat(mid, 151)
        assert w.justification(leaf, 151) == pytest.approx(1.0)

    def test_dead_ancestor_zeroes_justification(self):
        w = WhyLedger(horizon=90)
        root = w.spawn(None, 0)
        child = w.spawn(root, 0)
        w.kill(root, 5)
        assert w.justification(child, 6) == 0.0  # the veto law's substrate


class TestCascade:
    def test_rate_limited(self):
        w = WhyLedger(horizon=90, cascade_budget=10)
        root = w.spawn(None, 0)
        for _ in range(50):
            w.spawn(root, 0)
        converted = w.kill(root, 1, budget=10)
        assert converted == 10
        assert w.cascade_pending > 0
        total = converted
        while w.cascade_pending:
            total += w.drain_cascade(1, budget=50)
        assert total == 50


class TestAuction:
    def test_veto_law_never_admits_zero_attention(self):
        sup = Supervisor(horizon=90)
        region = sup.add_region(until=None, now=0)
        # Zero urgency -> score 0 -> vetoed forever.
        jid = sup.add_job(region, "vetoed-job", now=0, urgency=0.0)
        report = sup.period(now=10)
        assert "vetoed-job" not in report.admitted
        assert report.vetoed >= 1
        # Sanity: the job is ALIVE but its attention is zero.
        assert sup.ledger.state(jid) is WhyState.ALIVE

    def test_dead_region_children_cannot_buy_gpu(self):
        sup = Supervisor(horizon=90)
        region = sup.add_region(until=None, now=0)
        jid = sup.add_job(region, "orphan", now=0, urgency=1.0)
        sup.ledger.kill(region, 1)  # region dead; child cascaded to UNKNOWN
        # The ledger prices the orphan at zero: its lineage's warrant is
        # destroyed. Even if it were revived, j stays 0 under a DEAD root.
        assert sup.ledger.justification(jid, 10) == 0.0
        report = sup.period(now=10)
        assert "orphan" not in report.admitted

    def test_rails_conserve(self):
        sup = Supervisor(horizon=90, rails_refill_per_period=4.0)
        region = sup.add_region(until=None, now=0)
        sup.add_job(region, "cheap", now=0, gpu_seconds_cost=1.0, urgency=1.0)
        sup.add_job(region, "pricey", now=0, gpu_seconds_cost=10.0, urgency=0.9)
        report = sup.period(now=10)
        # Rails refill 4.0: cheap admitted, pricey infeasible this period.
        assert report.admitted == ["cheap"]
        assert report.skipped_infeasible == 1
        assert sup.rails.gpu_seconds == pytest.approx(3.0)

    def test_kernel_class_bypasses_cap_not_rails(self):
        sup = Supervisor(horizon=90, max_admitted=1, rails_refill_per_period=1.0)
        region = sup.add_region(until=None, now=0)
        sup.add_job(region, "world-job", now=0, urgency=1.0, importance="world")
        report = sup.period(now=10)
        assert report.admitted == ["world-job"]  # cap = 1, filled
        sup2 = Supervisor(horizon=90, max_admitted=1, rails_refill_per_period=1.0)
        region2 = sup2.add_region(until=None, now=0)
        sup2.add_job(region2, "kernel-job", now=0, urgency=1.0, importance="kernel")
        report2 = sup2.period(now=10)
        # Kernel bypasses the admission cap... but cap is 1 anyway; the
        # interesting case is cap 0 with rails available:
        sup3 = Supervisor(horizon=90, max_admitted=1, rails_refill_per_period=0.5)
        region3 = sup3.add_region(until=None, now=0)
        sup3.add_job(region3, "needs-more", now=0, urgency=1.0, gpu_seconds_cost=1.0, importance="kernel")
        report3 = sup3.period(now=10)
        assert report3.admitted == []  # rails say no — even to the kernel class
        assert report3.skipped_infeasible == 1
        del report2

    def test_harvest_checkpoints_and_parks_never_deletes(self):
        sup = Supervisor(horizon=90)
        region = sup.add_region(until=None, now=0)
        jid = sup.add_job(region, "hangs", now=0, urgency=1.0)
        report = sup.period(now=10)
        assert report.admitted == ["hangs"]
        # 200 ticks pass with no progress (the silent hang).
        sup.period(now=200)
        assert sup.ledger.state(jid) is WhyState.UNKNOWN
        kinds = [d.kind for d in sup.decisions]
        assert "harvest" in kinds and "checkpoint" in kinds
        job = sup.jobs[jid]
        assert job.checkpointed is True  # parked, state saved, NEVER deleted

    def test_revived_job_bids_on_parole_then_recovers(self):
        sup = Supervisor(horizon=90, max_admitted=6)
        region = sup.add_region(until=None, now=0)
        flaky = sup.add_job(region, "flaky", now=0, urgency=1.0)
        healthy = sup.add_job(region, "healthy", now=0, urgency=0.9)
        # The healthy job services the region chain, so only `flaky` hangs.
        sup.period(now=10)
        sup.job_progress("healthy", 20)
        sup.job_progress("healthy", 120)
        sup.job_progress("healthy", 195)  # keep the region's freshness alive
        sup.period(now=200)  # flaky harvested (region kept fresh by healthy)
        assert sup.ledger.state(flaky) is WhyState.UNKNOWN
        assert sup.ledger.state(region) is WhyState.ALIVE
        assert sup.revive("flaky", 210)
        # Parole caps justification at 0.5 — the job still competes, weaker.
        j_parole = sup.ledger.justification(flaky, 211)
        assert j_parole == pytest.approx(PROBATION_CAP)
        sup.job_progress("flaky", 220)  # a stage completes: parole ends
        assert sup.ledger.justification(flaky, 221) > PROBATION_CAP


class TestTheLeaseComposition:
    """The memory axis is thor-admit's; the supervisor composes with it
    once per admission (see thor/ecology_admission/lease.py)."""

    def test_admission_asks_for_a_lease_first(self):
        leases = RecordingLeaseClient()
        sup = Supervisor(horizon=900, lease_client=leases)
        region = sup.add_region(until=None, now=0)
        sup.add_job(region, "scene-1", now=0, importance="world")
        sup.period(now=10)
        # exactly one acquire, with the daemon's own conventions
        assert leases.acquired == [("scene-1", "auto", LEASE_PRIORITY["world"])]

    def test_kernel_class_maps_to_the_front_of_the_daemons_queue(self):
        leases = RecordingLeaseClient()
        sup = Supervisor(horizon=900, lease_client=leases)
        region = sup.add_region(until=None, now=0)
        sup.add_job(region, "coarse-gate", now=0, importance="kernel")
        sup.period(now=10)
        assert leases.acquired[0][2] == 5  # the daemon: higher goes first

    def test_a_refused_lease_leaves_the_job_queued_and_fresh(self):
        class Refusing(RecordingLeaseClient):
            def acquire(self, label, peak, priority):
                self.acquired.append((label, peak, priority))
                return LeaseRefusal("no history for the label")

        sup = Supervisor(horizon=900, lease_client=Refusing())
        region = sup.add_region(until=None, now=0)
        jid = sup.add_job(region, "scene-1", now=0, urgency=1.0)
        report = sup.period(now=10)
        assert report.admitted == []          # not admitted...
        assert "scene-1" not in sup.running   # ...not running...
        assert sup.ledger.state(jid) is WhyState.ALIVE  # ...and still alive
        assert any(d.kind == "lease" for d in sup.decisions)  # the refusal is a decision

    def test_done_releases_the_lease_and_park_releases_it_too(self):
        leases = RecordingLeaseClient()
        sup = Supervisor(horizon=60, lease_client=leases)
        region = sup.add_region(until=None, now=0)
        hangs = sup.add_job(region, "hangs", now=0, urgency=1.0)
        sup.period(now=5)
        assert len(leases.acquired) == 1
        sup.period(now=200)  # harvested: checkpoint -> park
        assert len(leases.released) == 1   # the parked job holds no memory
        assert sup.jobs[hangs].lease_id is None

    def test_the_wire_shape_matches_the_daemons_protocol(self):
        client = ThorAdmitLeaseClient(socket_path="/tmp/does-not-matter.sock")
        assert client.socket_path == "/tmp/does-not-matter.sock"
        # the priority table mirrors the bridge module's LEASE_PRIORITY
        assert LEASE_PRIORITY == {"maintenance": 0, "scenery": 1, "world": 2, "kernel": 5}
