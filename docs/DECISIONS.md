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

The production topology remains independent GPU lanes with a shared host expert arena. The current promoted engine generation is **Strata 0.1.39**, integrated from upstream `6f32ec070f23ced9f50e704d854d775da52591ab` through PR #24 and promoted on main at merge commit `9ae0839b9f376dca9742804924db05902f872ad8`. The validated sync head before promotion was `1a454ca80d698af863693c1b2d3fafac0b3b1ceb`; the performance binary used for the topology campaign was built from the same source generation before later docs-only commits. The promoted baseline preserves the Phase 1/2 serving-control contract and the Phase 3 leader/follower shared-arena lifecycle while consuming upstream 0.1.39 batching, Responses API, conversation-cache telemetry, decode/kernel, CPU-pool, expert-cache and low-RAM/file-tier work. Ordinary lanes explicitly strip inherited engine-internal multi-GPU flags plus `parallel` / `--batch` / `--slots` / `--batch-groups` / `--trim-stage-weights`, and shared `--expert-profile-save` writers, so only explicit challenger contracts can re-enable nested execution. The 0.1.39 promotion gate passed the full Python serving suite (**333 tests, 7 skipped**), fresh CUDA 13.4 sm_120 Release builds with tests both off and on, focused `pinned_shared_test`, `expert_profile_save_test`, and `file_expert_source_test` CTests, a live single-lane model smoke, a live three-lane shared-arena concurrent-generation smoke, and a live Responses API request through the Lanes supervisor. Historical 0.1.30/0.1.31/0.1.38 measurements retain their original engine labels. Hardware-specific measurements and caveats live in the public recipe repository.

The 2026-10-05 reference-host topology recheck changes one challenger conclusion without changing the promoted topology. The fixed upstream-native three-GPU layer-split pipeline (`--layer-split 18,34 --batch 3 --batch-groups 3 --trim-stage-weights`, shared expert arena) measured **147.84 ± 2.26 tok/s** at M=2 versus **143.19 ± 3.06** for two independent lanes (**+3.25%**) and **209.66 ± 4.02 tok/s** at M=3 versus **192.16 ± 4.11** for three lanes (**+9.11%**). The retained same-binary single-request layer-split control is **120.62 ± 1.24 tok/s** versus **73.63 ± 1.67** for one lane, but M=1 was not rerun under the exact batch3/groups3/trim config. Cold-prefill still crosses by workload length: ~15K x3 strongly favors lanes (**5901.34 vs 3289.19 tok/s**), while ~110K x3 narrowly favors layer split (**6028.09 vs 5822.71 tok/s**). Every retained PP request reported `cache_n=0`; the nonce text is workload/length matched rather than byte-identical across topology arms. The ordinary FIFO path is now treated only as a serial control. This evidence reopens static layer split / Super-Lane as a serious topology candidate but does not by itself change the production default. See `docs/strata-0.1.39-performance-crossover-20261005.md`.

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
