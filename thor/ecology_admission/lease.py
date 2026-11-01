"""lease.py — the memory axis is thor-admit's; this is how we speak to it.

The landscape changed after this proposal was first written (2026-11-01):
builder-ide's ``thor-admit`` branch built the box's memory admission
daemon — kernel-held leases a dead client cannot leak, a 250 ms watchdog
that stops the largest lease long before the kernel OOMs, and learned
peaks (the p95 of a label's history) that feed the cost model's memory
column.

The ecology supervisor does not duplicate that. The split is clean:

  * **thor-admit owns memory**: does the box have room for this peak?
  * **the ecology owns purpose**: is this job worth running, is it
    healthy (freshness), and what happens when it stalls
    (UNKNOWN + checkpoint, never delete)?

They compose once per admission: the auction admits a job, the
supervisor asks thor-admit for the lease, the lease is granted, the
process spawns. A lease refused by the daemon is not a scheduler
failure — the job stays queued and its purpose stays fresh (it is not
the job's fault the box is full).

This module is the client half of that composition, in three sizes:

  * :class:`NullLeaseClient` — the dry-run default. Grants everything,
    journals nothing to any daemon. The simulator uses this.
  * :class:`ThorAdmitLeaseClient` — speaks the daemon's JSON protocol
    over its unix socket (its design doc §4). No dependencies beyond
    the stdlib.
  * :class:`RecordingLeaseClient` — grants everything and records what
    it was asked, for tests and for shadow mode ("what *would* we have
    leased?").
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Importance name -> thor-admit queue priority (higher goes first).
#: Mirrors the bridge module's LEASE_PRIORITY (builder-ide
#: research/ecology-bridge, core/ecology_bridge/auction.py).
LEASE_PRIORITY = {"maintenance": 0, "scenery": 1, "world": 2, "kernel": 5}


@dataclass(frozen=True)
class LeaseGrant:
    """What the daemon hands back: the lease id and the peak it booked."""

    lease_id: int
    peak_bytes: int
    lock_path: str = ""

    @property
    def ok(self) -> bool:
        return self.lease_id >= 0


@dataclass
class LeaseRefusal:
    reason: str

    @property
    def ok(self) -> bool:
        return False


class LeaseClient:
    """The protocol the supervisor speaks for the memory axis."""

    def acquire(self, label: str, peak, priority: int) -> LeaseGrant | LeaseRefusal:
        raise NotImplementedError

    def release(self, lease_id: int, peak_rss_bytes: int = 0) -> bool:
        raise NotImplementedError


class NullLeaseClient(LeaseClient):
    """Dry-run: memory is not our problem tonight."""

    _next = 0

    def acquire(self, label: str, peak, priority: int) -> LeaseGrant:
        type(self)._next += 1
        return LeaseGrant(lease_id=type(self)._next, peak_bytes=0)

    def release(self, lease_id: int, peak_rss_bytes: int = 0) -> bool:
        return True


class RecordingLeaseClient(LeaseClient):
    """Grants everything, remembers everything — shadow mode and tests."""

    def __init__(self) -> None:
        self.acquired: List[Tuple[str, object, int]] = []
        self.released: List[int] = []

    def acquire(self, label: str, peak, priority: int) -> LeaseGrant:
        self.acquired.append((label, peak, priority))
        return LeaseGrant(lease_id=len(self.acquired), peak_bytes=0)

    def release(self, lease_id: int, peak_rss_bytes: int = 0) -> bool:
        self.released.append(lease_id)
        return True


class ThorAdmitLeaseClient(LeaseClient):
    """The daemon's own protocol, over its unix socket.

    The wire is JSON, one object per line (the daemon's design §4):
    ``{"op":"acquire","label":...,"peak":...,"priority":...,
    "target":{"group":{"pgid":...}}}`` and back
    ``{"event":"granted","lease":...,"peak_bytes":...,"lock_path":...}``.
    ``peak="auto"`` asks for the learned p95 of the label's history —
    which the daemon refuses for a label it has never seen; on that
    refusal the caller falls back to the cost model's memory column
    (never to a guess).
    """

    def __init__(self, socket_path: Optional[str] = None) -> None:
        import os

        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/tmp")
        self.socket_path = socket_path or os.path.join(
            runtime, f"thor-admit-{os.getuid()}.sock"
        )
        if Path(self.socket_path).name == "thor-admit.sock":
            # the daemon's documented default puts it under the runtime dir
            self.socket_path = os.path.join(runtime, "thor-admit.sock")

    def _ask(self, request: Dict) -> Dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            sock.sendall((json.dumps(request) + "\n").encode())
            buffer = b""
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            return json.loads(buffer.decode() or "{}")

    def acquire(self, label: str, peak, priority: int,
                pgid: Optional[int] = None) -> LeaseGrant | LeaseRefusal:
        target = {"group": {"pgid": pgid}} if pgid is not None else {"group": {}}
        answer = self._ask(
            {"op": "acquire", "label": label, "peak": peak,
             "priority": priority, "target": target}
        )
        if answer.get("event") == "granted":
            return LeaseGrant(
                lease_id=int(answer["lease"]),
                peak_bytes=int(answer.get("peak_bytes", 0)),
                lock_path=str(answer.get("lock_path", "")),
            )
        return LeaseRefusal(str(answer.get("reason", answer.get("event", "no lease"))))

    def release(self, lease_id: int, peak_rss_bytes: int = 0) -> bool:
        answer = self._ask(
            {"op": "release", "lease": lease_id, "peak_rss_bytes": peak_rss_bytes}
        )
        return answer.get("event") == "ok"
