# Architecture decision provenance

This fork uses **decision-bearing GitHub Issues as ADR-style decision records**.

The goal is to keep architectural reasoning close to the work that proposes and validates it without creating a second, competing lifecycle for standalone ADR files. Canonical documentation, source, and tests remain the source of truth for the current implemented contract; Issues preserve decision provenance and discussion.

## When an Issue is decision-bearing

Use the decision-bearing format when an Issue proposes or changes one or more of the following:

- architecture or major runtime structure;
- API, protocol, schema, or compatibility behavior;
- ownership or source-of-truth boundaries;
- migration or fallback policy;
- security, safety, or failure-isolation boundaries;
- cross-component contracts;
- durable contributor or operations workflow.

Routine bugs, small implementation tasks, benchmark runs, and investigations do not need ADR-style ceremony unless they also make one of those durable decisions.

## Required sections

A decision-bearing Issue should contain these sections.

### Decision status

Use ordinary prose such as:

- **Proposed** — under consideration; not yet authoritative.
- **Accepted** — explicitly accepted by the maintainer or already established by the canonical contract.
- **Superseded** — replaced by another Issue or canonical contract; link the replacement.

Do not infer acceptance merely because an Issue was opened or implementation work started.

### Context / problem

Describe the concrete problem, current architecture, workload, or constraint that motivates the decision. Link the current canonical document and relevant implementation when they already exist.

### Proposed decision / contract

State the exact architectural or behavioral delta being proposed. Separate current behavior from proposed behavior.

### Rationale / evidence

Record measurements, reproductions, source evidence, operational observations, or other reasons supporting the proposal. Distinguish measured facts from expectations that still need validation.

### Alternatives considered or rejected

Record meaningful alternatives and why they are not the current proposal. An alternative can remain a future experimental challenger instead of being permanently rejected.

### Consequences / trade-offs / non-goals

Document new coupling, complexity, resource cost, compatibility implications, failure modes, and what the decision explicitly does not attempt to solve.

### Acceptance / validation

Define how the proposal will be judged. Prefer measurable gates and real workload validation over aesthetic or implementation-only completion criteria.

## Source-of-truth rule

A decision-bearing Issue is **decision provenance**, not automatically the live contract.

For this fork, current source-of-truth precedence is:

1. current implementation and tests for executable behavior;
2. canonical repository documentation for the intended durable contract;
3. accepted decision-bearing Issues and their linked implementation history;
4. proposed or superseded Issues as historical context only.

When an accepted decision changes durable behavior, update the owning canonical documentation with the implementation and link the relevant Issue/PR.

## Multi-GPU decisions

The current canonical multi-GPU documents are:

- [`multigpu-shared-runtime.md`](multigpu-shared-runtime.md) — implemented shared-arena / independent-lane runtime contract;
- [`multigpu-hardware-guide.md`](multigpu-hardware-guide.md) — hardware sizing, RAM/CPU/PCIe guidance and bring-up checklist;
- [`multigpu-roadmap.md`](multigpu-roadmap.md) — future architecture challengers and promotion gates.

Concrete reference-host hardware, tuning values, benchmark numbers, and production validation records intentionally live in the separate public recipe repository: [`rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe`](https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe).

The production baseline remains independent GPU lanes with a shared host expert arena. The current promoted engine generation is **Strata 0.1.38**, integrated from upstream `99f3dbd0b21d1401b3769e0c0d963913607f380b`. The next sync candidate is **Strata 0.1.39**, merged on `sync/upstream-0.1.39` at `3727c6bdfb12ba2ffc07402b68ae86c0ed2ac27f` from upstream `6f32ec070f23ced9f50e704d854d775da52591ab`; it is validated but not promoted until main moves. The candidate preserves the Phase 1/2 serving-control contract and the Phase 3 leader/follower shared-arena lifecycle while consuming upstream 0.1.39 batching, Responses API, conversation-cache telemetry, decode/kernel, CPU-pool, expert-cache and low-RAM/file-tier work. Ordinary lanes explicitly strip inherited engine-internal multi-GPU flags plus `parallel` / `--batch` / `--slots` / `--batch-groups` / `--trim-stage-weights`, and shared `--expert-profile-save` writers, so only future explicit contracts can re-enable nested execution. The 0.1.39 bounded gate passed the full Python serving suite (**333 tests, 7 skipped**), fresh CUDA 13.4 sm_120 Release builds with tests both off and on, focused `pinned_shared_test`, `expert_profile_save_test`, and `file_expert_source_test` CTests, a live single-lane model smoke, and a live three-lane shared-arena concurrent-generation smoke; follower lanes logged that the shared population was ready and skipped the repeated source expert load. The earlier 0.1.38 matched long-prompt compatibility A/B against the installed 0.1.31 binary remains historical evidence at roughly +9.5% for ~15K tokens and +7.3% for ~30K tokens; it is not relabeled as 0.1.39 performance evidence. Upstream layer-split/peer/batch execution remains complementary to the fork's independent request/session lanes. Hardware-specific measurements and caveats live in the public recipe repository.

The 2026-10-05 reference-host performance recheck changes one challenger conclusion without changing the promoted topology. On the 0.1.39 candidate, three independent lanes measured **192.16 ± 4.11 tok/s** common-wall aggregate TG for three fixed 512-token reasoning requests. The ordinary layer-split FIFO path is only a serial control (**118.75 ± 5.98 tok/s**), while upstream 0.1.39's explicit three-stage pipeline (`--layer-split 18,34 --batch 3 --batch-groups 3 --trim-stage-weights`, shared expert arena) reached **209.66 ± 4.02 tok/s**, **9.1% above** the independent-lane arm in that fixed decode region. Cold-prefill A/Bs show a workload-length crossover rather than a universal winner: at ~15K x3, independent lanes reached **5901.34 ± 55.16 tok/s** versus **3289.19 ± 18.83** for the pipelined split; at ~110K x3, the pipelined split reached **6028.09 ± 14.50** versus **5822.71 ± 4.49** for lanes. Every retained PP request reported `cache_n=0`. The medium/long PP nonce text is contract-matched rather than byte-identical across arms. This evidence reopens static layer-split/Super-Lane work as a measured workload-dependent challenger, but does not promote it over independent lanes or justify automatic topology changes. See `docs/strata-0.1.39-performance-crossover-20261005.md`.

## Template for a decision-bearing Issue

```markdown
## Decision status

Proposed

## Context / problem

What exists today, what problem is being solved, and which canonical contract currently owns this behavior?

## Proposed decision / contract

What exact durable change is proposed?

## Rationale / evidence

What measurements, source evidence, or operational observations support it?

## Alternatives considered or rejected

What other approaches were considered, and why are they not the current proposal?

## Consequences / trade-offs / non-goals

What gets more complex, more coupled, more expensive, or explicitly remains out of scope?

## Acceptance / validation

What reproducible conditions must be true before this decision can be accepted or promoted?
```
