"""ecology_admission — attention-weighted admission control for infinigen-thor.

A standalone prototype of the ecology-thor supervisor (see
thor/proposals/ecology-admission.md). Pure Python; transparently uses the
compiled ecology_thor wheel when present, otherwise the local mirror.
"""

from .ledger import PROBATION_CAP, WhyLedger, WhyState, link_freshness
from .supervisor import AdmissionReport, Decision, Job, Rails, Supervisor

__all__ = [
    "PROBATION_CAP",
    "WhyLedger",
    "WhyState",
    "link_freshness",
    "AdmissionReport",
    "Decision",
    "Job",
    "Rails",
    "Supervisor",
]
