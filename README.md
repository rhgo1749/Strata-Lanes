# Strata-Lanes

> **Research fork / experimental evidence archive.**
>
> This repository is **not recommended as the general-purpose way to run Strata**.
> For normal installation and day-to-day Strata use, use the upstream project:
> **[Niko1221/Strata](https://github.com/Niko1221/Strata)**.

Strata-Lanes started as an experiment in **request-level GPU parallelism**: run one independent whole-model Strata engine per GPU, keep session/KV state lane-local, and physically share the large host-RAM expert arena across the processes.

The implementation is kept here because the experiment produced useful systems evidence: shared-memory accounting, independent-lane scaling, session-affinity scheduling, queue behavior, conversation parking, heterogeneous-GPU isolation, and a sequence of matched comparisons against upstream multi-GPU execution.

## Why this is now a research record

The original motivation was that independent GPU lanes could preserve single-GPU execution while scaling aggregate serving throughput without NVLink or token-by-token cross-GPU synchronization.

That was true in important workload regions, but upstream Strata changed materially.

With Strata 0.1.39, upstream added batch slots, layer-split pipeline groups, and stage-weight trimming. On the reference 3×RTX 5070 Ti host, the strongest upstream-native three-GPU layer-split configuration now matches or exceeds independent lanes in several regions.

| Region | Independent lanes | Pipelined layer split |
| --- | ---: | ---: |
| M=1 fixed decode | 73.63 ± 1.67 tok/s | **120.62 ± 1.24 tok/s**¹ |
| M=2 fixed decode | 143.19 ± 3.06 tok/s | **147.84 ± 2.26 tok/s** |
| M=3 fixed decode | 192.16 ± 4.11 tok/s | **209.66 ± 4.02 tok/s** |
| three ~15K cold prompts | **5901.34 ± 55.16 tok/s** | 3289.19 ± 18.83 tok/s |
| three ~110K cold prompts | 5822.71 ± 4.49 tok/s | **6028.09 ± 14.50 tok/s** |

¹ M=1 is the retained same-binary three-GPU layer-split single-request control; the exact batch3/groups3/trim configuration was not rerun at M=1.

So the current evidence is a **workload-dependent topology crossover**, not “lanes always win concurrency” and not “layer split always wins.”

Because upstream is now the better default place for normal users, this fork no longer presents independent lanes as a generally recommended deployment architecture.

## What is preserved here

This repository remains useful as an active research/evidence archive for:

- independent whole-model GPU lanes;
- a shared file-backed host expert arena across lane processes;
- leader/follower arena population and lifecycle;
- session affinity and lane-local state ownership;
- queue/admission instrumentation;
- conversation parking experiments;
- heterogeneous-GPU serving and interference studies;
- historical decode-assist / Super-Lane experiments;
- matched independent-lane ↔ upstream layer-split comparisons;
- paper/revision evidence and source provenance.

The code is intentionally preserved so old measurements can be reproduced and future upstream changes can be compared against the same experimental architecture.

## Current evidence

The compact 0.1.39 topology record is:

- [0.1.39 topology crossover](docs/strata-0.1.39-performance-crossover-20261005.md)
- [Paper v2 evidence snapshot](docs/paper-v2-evidence-20261005.md)
- public retained raw data and harnesses in the companion evidence repository:
  [rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe](https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe)

Canonical implementation history and decisions:

- [Architecture decisions](docs/DECISIONS.md)
- [Shared runtime contract](docs/multigpu-shared-runtime.md)
- [Multi-GPU roadmap](docs/multigpu-roadmap.md)
- [Fork / implementation boundary](docs/fork-and-implementation.md)

## Reproducing the historical Lanes runtime

The multi-lane implementation still exists in this fork and can be used for controlled reproduction.

See [Strata-Lanes multi-lane usage](docs/LANES_USAGE.md).

That document is now **reproduction/experimental documentation**, not a recommendation to replace upstream Strata for normal serving.

The central runtime shape is still:

    client requests
          |
    session-aware supervisor
       /    |    \
    lane0  lane1  lane2       <- independent Strata engines
       \     |     /
     shared host expert arena

Each lane owns its own CUDA state, expert cache, KV/session state, speculative state, and generation loop. The host expert mapping can be physically shared.

## Repository status

- **Mode:** active research / evidence archive
- **Upstream for normal use:** [Niko1221/Strata](https://github.com/Niko1221/Strata)
- **Current imported engine generation:** Strata 0.1.39
- **General deployment recommendation:** none
- **Historical Lanes implementation:** preserved and reproducible
- **Future changes:** evidence-driven experiments, provenance fixes, or upstream-comparison work

This is intentionally **not** GitHub-archived: the evidence may still be extended if upstream behavior changes or a paper revision needs additional validation.

## Paper evidence

- paper-v1 identifies the submitted v1 state.
- paper-v2-evidence identifies the current post-v1 experimental evidence snapshot.
- paper-v2 remains reserved for an actual revised manuscript/submission state.

The latest evidence manifest is [docs/paper-v2-evidence-20261005.md](docs/paper-v2-evidence-20261005.md).

## Credits and license

Upstream Strata is created and maintained by **Niko1221**. This fork inherits Strata's codebase and carries the experimental Lanes extensions and evidence history described above.

Strata and this fork are distributed under the licenses present in the repository. Model files and third-party components retain their own licenses.
