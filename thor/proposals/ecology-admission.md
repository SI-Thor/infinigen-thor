# Proposal: ecology-admission — attention-weighted admission control and UNKNOWN harvesting for infinigen-thor

*From the ecology-thor team (SI-Thor/ecology-thor, private for now).
Written for the infinigen-thor builders. Nothing here is merged intent —
it is a specific, tested proposal plus a running prototype, for your
review. Everything below runs standalone (no Rust build required).*

## The problem, in your own numbers

The spike (`thor/SPIKE.md`) and harvest branch established:

- **Renders are the GPU workload** (98% util, ~3 GB, minutes each), coarse
  is CPU-serial at ~2/14 cores, scenes are embarrassingly parallel.
- **The dominant overnight failure is the silent hang** (the EXACT-
  Boolean class you fixed to FAST, but the long tail is real): a job
  stops making progress, holds its RAM and GPU slot, and nothing ever
  decides it is dead.
- **V1 job states have no timeout and no UNKNOWN**:
  `notqueued/queued/running/succeeded/crashed/cancelled`. A hang is a
  `running` that lies forever.
- You already built the *checkpoint primitive*: `--on_interrupt`
  (SIGTERM/SIGUSR1 → `save_blend`) — state saved, not lost.
- Admission is static: `num_concurrent ≈ RAM/20GB` from the docs' rule,
  though your measurements show the rule is ~6× conservative.

That is precisely the shape the parent project's laws were built for:
purpose as a **subscription** (not a purchase), stalls priced by
freshness *before* anyone declares death, and revival as an explicit
ceremony.

## The proposal

Adopt the **ecology-thor kernel** (SI-Thor/ecology-thor; private — ask
Casey for access) as an *external supervisor* over infinigen job
execution. Three changes, in adoption order, each valuable alone:

### 1. Heartbeat + UNKNOWN harvesting (kills the silent hang class)

- Every running scene job maps to a **purpose node** in a WHY ledger
  (region goals are parents; scenes are children).
- Progress events re-attest: a job that logs/advances (your parseable
  `[HH:MM:SS.mmm] [module] [LEVEL] | msg` lines, or `metadata.json`
  generator-stage timings) stamps `run_under` — service re-attests the
  whole chain leaf→root.
- No progress for longer than an **attestation horizon** → the job's
  purpose *lapses to UNKNOWN* — "questionable, not gone". The supervisor
  then executes the policy you already prefer: SIGUSR1 → `--on_interrupt`
  checkpoint → park the job (never delete: output dir and checkpoint stay).
- Revival is explicit (`revive`, on parole — the re-attested job bids at
  half strength until it completes a stage, which ends parole). A root
  goal that misses its deadline dies and its **living descendants cascade
  to UNKNOWN** — the whole region becomes questionable in one visible
  move, rate-limited so the supervisor never thrashes.

Key property vs. a plain timeout: **freshness slides before lapse**. A
job's justification decays continuously (grace window, then linear to
zero at the horizon), so a stalling job *loses admission priority to
healthy jobs before it is ever killed*. You get graceful degradation
for free; the timeout is just the end of the slide.

### 2. Attention-weighted admission (replaces static `num_concurrent`)

The box's admission budget becomes **rails**: `gpu_seconds` (and joules)
refilled per scheduling period. Candidate scene jobs bid:

```
score = urgency · justification · opportunity · novelty · expected_value
```

- `urgency`: deadline pressure of the region goal;
- `justification`: the chain freshness above (healthy lineage, fresh beats);
- `opportunity`: GPU/RAM headroom right now (your measured ~3 GB/render
  vs 122 GB means the real concurrent-render ceiling is memory and
  inter-render interference, not the docs' 20 GB rule — the auction
  discovers it empirically);
- `novelty`: new-seed scenes over re-runs;
- `expected_value`: your p50 stage timings per scene type.

Veto law: score ≤ 0 never admits — a DEAD region's children (justification
0) cannot buy GPU-seconds no matter their urgency. Kernel-class work
(coarse passes that gate everything) bypasses the admission cap but not
the rails.

### 3. Splice memory (the throughput multiplier)

Your `mavis/thor-analysis` subtree-memo (2.33× edited-scene regen,
98/128 builders skipped) is, in our vocabulary, a **novelty/expected_value
oracle**: a recipe-graph hash that says "this goal was served before".
Wiring the memo into the bid's `expected_value` (a memo-hit bids high
novelty-of-completion at low cost) is the highest-leverage throughput
change on the box — and it is a pure bid-policy change, no kernel edit.

## What we built for you (on this branch)

`thor/ecology_admission/` — a **standalone prototype supervisor**:

- `ledger.py`: the WHY ledger semantics (four states, chain freshness,
  lapse, parole, cascades) in pure Python — a faithful mirror of the Rust
  kernel's math, so you can dogfood the *semantics* before deciding to
  build the Rust wheel.
- `supervisor.py`: job model + admission policy (heartbeat from job
  events, rails, SIGUSR1-before-kill, checkpoint parking, revive).
- `sim.py`: a dry-run, fake-clock simulator of an overnight batch
  (stalls included) that prints the decisions the supervisor *would*
  make — `python -m thor.ecology_admission.sim`.
- `tests/test_ecology_admission.py`: pytest, fake clock, no infinigen
  import — CI-clean.

If you later `pip install` the compiled kernel wheel
(`ecology-thor`, built by `ecology-py`), `ledger.py` transparently
delegates to it (`try: import ecology_thor`) — same semantics,
microseconds instead of milliseconds, 100k-job arenas.

## What we are NOT proposing

- No changes to infinigen core, no new process inside your render path.
  The supervisor is external; jobs stay exactly as they are.
- No deletion of anything, ever (outputs, checkpoints, states). The
  UNKNOWN state is the whole point: questionable, not gone.
- No re-write of `manage_jobs.py` — the prototype drives a dry-run model;
  the real integration point is `LocalScheduleHandler` /
  `monitor_tasks.py`, and it stays yours.

## Adoption path

1. Run the sim (2 min): `python -m thor.ecology_admission.sim --ticks 3600`.
2. Argue with the semantics in review — especially horizon length
   (we default: 3× your p95 render time) and parole rules.
3. Shadow mode over a real `seed-sweep.sh` batch: supervisor logs
   decisions, executes nothing. Compare overnight throughput and hang
   latency vs. status quo.
4. Flip to enforcing mode for one batch kind. Measure. Keep or revert —
   the supervisor is one flag away from off.

## Questions we want your review on

1. Is SIGUSR1→checkpoint→park acceptable for *coarse* stage jobs
   (mid-stage state is larger), or should coarse jobs get a longer
   horizon instead of checkpoints?
2. Should region-goal deadlines (`until`) come from your gin configs,
   the night runner, or stay manual in v1?
3. The rails' `gpu_seconds` refill rate — we suggest deriving it from
   your measured p50 render time × the concurrency you actually want;
   better ideas welcome.

— the ecology-thor team
