"""WHY ledger semantics — a faithful pure-Python mirror of ecology-thor.

This module exists so the infinigen-thor team can dogfood the *semantics*
of the ecology-thor kernel (SI-Thor/ecology-thor) before building any Rust
wheel. Every formula here mirrors `crates/ecology-core/src/why.rs` and
`freshness.rs` line-for-line:

- four states, never deleted (there is no delete API at all);
- link freshness: full through horizon/3 (grace), then linear to 0 at the
  horizon; stale information never rewinds freshness;
- chain justification: alive-fraction of the root path x weakest-link
  freshness, zeroed by any DEAD ancestor, capped at PROBATION_CAP while
  any ALIVE link is on parole;
- lapse: unattested past the horizon -> UNKNOWN (per node, no cascade);
- revive: UNKNOWN -> ALIVE on parole; parole ends on service;
- kill: DEAD + living descendants cascade to UNKNOWN (rate-limited).

If the compiled wheel is importable we delegate to it (identical
semantics, population scale); otherwise this implementation stands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - wheel is optional
    import ecology_thor as _rust  # type: ignore

    _HAVE_RUST = True
except Exception:  # noqa: BLE001 - any import failure means: standalone
    _HAVE_RUST = False

PROBATION_CAP = 0.5


class WhyState(Enum):
    """The four states of purpose (byte values match the kernel)."""

    ABSENT = 0
    ALIVE = 1
    UNKNOWN = 2
    DEAD = 3


class SpawnState(Enum):
    ALIVE = "alive"
    ABSENT = "absent"


@dataclass
class Node:
    index: int
    parent: Optional[int]
    born: int
    state: WhyState
    last_beat: int  # -1 until the first beat
    beat_count: int = 0
    until: Optional[int] = None  # root deadline (ticks); None = immortal
    probation: bool = False
    mass: float = 1.0


def link_freshness(state: WhyState, last_beat: int, born: int, now: int, horizon: int) -> float:
    """Freshness of one ALIVE link (kernel `freshness.rs::link_freshness`)."""
    if state is not WhyState.ALIVE:
        return 1.0  # non-alive links don't reduce the path minimum
    last = last_beat if last_beat >= 0 else born
    age = max(0, now - last)
    grace = horizon // 3
    if age <= grace:
        return 1.0
    if age >= horizon:
        return 0.0
    return 1.0 - (age - grace) / (horizon - grace)


class WhyLedger:
    """Trees, never DAGs (same restriction as the kernel, same reason:
    the chain aggregate is one upward walk)."""

    def __init__(self, horizon: int, deadline_grace: int = 30, cascade_budget: int = 64):
        if horizon < 1:
            raise ValueError("horizon must be >= 1 tick")
        self.horizon = horizon
        self.deadline_grace = deadline_grace
        self.cascade_budget = cascade_budget
        self.nodes: List[Node] = []
        self._cascade_queue: List[int] = []
        self.transitions: List[Tuple[int, WhyState, WhyState, str, int]] = []

    # ------------------------------------------------------------- spawn

    def spawn(
        self,
        parent: Optional[int],
        now: int,
        mass: float = 1.0,
        until: Optional[int] = None,
        initial: SpawnState = SpawnState.ALIVE,
        reason: str = "born",
    ) -> int:
        if parent is not None and not (0 <= parent < len(self.nodes)):
            raise IndexError("no such parent")
        state = WhyState.ALIVE if initial is SpawnState.ALIVE else WhyState.ABSENT
        node = Node(
            index=len(self.nodes),
            parent=parent,
            born=now,
            state=state,
            last_beat=now if state is WhyState.ALIVE else -1,
            until=until,
            mass=mass,
        )
        self.nodes.append(node)
        self._record(node, state, reason, now, WhyState.ABSENT)
        return node.index

    # -------------------------------------------------------- transitions

    def _record(self, node: Node, to: WhyState, reason: str, now: int, from_state: WhyState) -> None:
        if from_state is to:
            return
        if to is WhyState.DEAD:
            node.mass = 0.0
        node.state = to
        self.transitions.append((node.index, from_state, to, reason, now))

    def state(self, index: int) -> WhyState:
        return self.nodes[index].state

    # ------------------------------------------------------- attestation

    def heartbeat(self, index: int, now: int) -> bool:
        """One node. Stale beats are no-ops (never rewind freshness)."""
        node = self.nodes[index]
        if node.state is not WhyState.ALIVE:
            return False
        if now <= node.last_beat:
            return False
        node.last_beat = now
        node.beat_count += 1
        node.probation = False  # service ends parole
        return True

    def run_under(self, index: int, now: int) -> int:
        """Service: stamp this node and every ALIVE ancestor to the root."""
        stamped = 0
        cur: Optional[int] = index
        while cur is not None:
            node = self.nodes[cur]
            if node.state is WhyState.ALIVE and now > node.last_beat:
                node.last_beat = now
                node.beat_count += 1
                node.probation = False
                stamped += 1
            cur = node.parent
        return stamped

    # ----------------------------------------------------------- lifecycle

    def kill(self, index: int, now: int, reason: str = "killed", budget: Optional[int] = None) -> int:
        """DEAD + living descendants cascade to UNKNOWN (rate-limited)."""
        budget = self.cascade_budget if budget is None else budget
        node = self.nodes[index]
        for child in self._children(index):
            self._cascade_queue.append(child)
        before = node.state
        self._record(node, WhyState.DEAD, reason, now, before)
        return self.drain_cascade(now, budget)

    def drain_cascade(self, now: int, budget: Optional[int] = None) -> int:
        budget = self.cascade_budget if budget is None else budget
        converted = 0
        while converted < budget and self._cascade_queue:
            idx = self._cascade_queue.pop()
            node = self.nodes[idx]
            if node.state is WhyState.ALIVE:
                self._record(node, WhyState.UNKNOWN, "cascade", now, node.state)
                converted += 1
            for child in self._children(idx):
                self._cascade_queue.append(child)
        return converted

    @property
    def cascade_pending(self) -> int:
        return len(self._cascade_queue)

    def revive(self, index: int, now: int, reason: str = "revived") -> bool:
        node = self.nodes[index]
        if node.state is not WhyState.UNKNOWN:
            return False
        node.probation = True
        node.last_beat = now
        self._record(node, WhyState.ALIVE, reason, now, WhyState.UNKNOWN)
        return True

    def reap(self, now: int) -> List[int]:
        """The economic harvest: unattested ALIVE nodes lapse to UNKNOWN."""
        lapsed = []
        for node in self.nodes:
            if node.state is not WhyState.ALIVE:
                continue
            last = node.last_beat if node.last_beat >= 0 else node.born
            if now - last > self.horizon:
                self._record(node, WhyState.UNKNOWN, "lapsed", now, WhyState.ALIVE)
                lapsed.append(node.index)
        return lapsed

    def deadline_check(self, now: int) -> List[int]:
        """Living roots past until+grace die (cascading, budgeted)."""
        died = []
        for node in self.nodes:
            if node.parent is not None or node.state is not WhyState.ALIVE:
                continue
            if node.until is not None and now - node.until > self.deadline_grace:
                self.kill(node.index, now, reason="deadline")
                died.append(node.index)
        return died

    # --------------------------------------------------------- aggregates

    def justification(self, index: int, now: int) -> float:
        """Alive-fraction x weakest-link freshness along the root path."""
        node = self.nodes[index]
        alive = 0
        length = 0
        worst = 1.0
        dead = False
        probation = False
        cur: Optional[int] = index
        while cur is not None:
            n = self.nodes[cur]
            length += 1
            is_alive = n.state is WhyState.ALIVE
            if is_alive:
                alive += 1
                fresh = link_freshness(n.state, n.last_beat, n.born, now, self.horizon)
                worst = min(worst, fresh)
                probation = probation or n.probation
            if n.state is WhyState.DEAD:
                dead = True
            cur = n.parent
        del node
        if dead or alive == 0:
            return 0.0
        score = (alive / length) * worst
        if probation:
            score = min(score, PROBATION_CAP)
        return score

    # ------------------------------------------------------------ helpers

    def _children(self, index: int) -> List[int]:
        return [n.index for n in self.nodes if n.parent == index]

    def counts(self) -> Dict[str, int]:
        out = {"alive": 0, "dead": 0, "unknown": 0, "absent": 0}
        for n in self.nodes:
            out[n.state.name.lower()] += 1
        return out
