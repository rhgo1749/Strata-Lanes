# Strata 0.1.39 topology crossover evidence — 2026-10-05

## Status

This record is part of the **promoted Strata 0.1.39 software baseline**. The measurements were collected on the validated pre-promotion source generation and remain the evidence for the current topology decision; concrete raw evidence and supporting controls live in the public recipe repository.

Measured source: `d51d7e9cbc327f90c2fd59b1e20a949033f594e2` on `sync/upstream-0.1.39`, upstream Strata `6f32ec070f23ced9f50e704d854d775da52591ab` (v0.1.39).

## Decode concurrency

The strongest measured three-GPU upstream-native challenger uses explicit split `18,34`, `--batch 3 --batch-groups 3 --trim-stage-weights`, and the shared expert arena.

| Concurrent requests | Independent lanes | Three-GPU layer split | Result |
| ---: | ---: | ---: | --- |
| 1 | 73.63 ± 1.67 tok/s | **120.62 ± 1.24 tok/s**¹ | layer split +63.8% |
| 2 | 143.19 ± 3.06 tok/s | **147.84 ± 2.26 tok/s** | layer split +3.25% |
| 3 | 192.16 ± 4.11 tok/s | **209.66 ± 4.02 tok/s** | layer split +9.11% |

¹ M=1 is the retained same-binary three-GPU layer-split single-request control. The exact batch3/groups3/trim config was not rerun at M=1. M=2 and M=3 are exact measurements of the fixed pipelined config.

The M=2 engine remained in `strata batch (pipelined, 3 groups of 1)`; with only two active requests one group is unused. Both requests completed together at roughly 6.9 s mean E2E, not as a FIFO staircase.

The old three-request FIFO result is therefore only a serial control, not the 0.1.39 layer-split concurrency ceiling.

## Cold-prefill crossover

Three cold prompts were submitted simultaneously and every retained request reported `cache_n=0`.

| Prompt regime | Independent lanes | Pipelined layer split | Result |
| --- | ---: | ---: | --- |
| ~15K x3 | **5901.34 ± 55.16 tok/s** | 3289.19 ± 18.83 | lanes +79.4% |
| ~110K x3 | 5822.71 ± 4.49 | **6028.09 ± 14.50** | layer split +3.5% |

The ~15K region favors independent request-level prefill. At ~110K the intra-prompt chunk pipeline is saturated enough to cross back slightly in favor of layer split.

The 15K/110K arms are workload/length matched but not byte-identical because the unique nonce encoded the topology arm.

## Decision impact

The 0.1.39 evidence no longer supports treating independent lanes as the general performance winner for concurrent decode. The fixed three-GPU layer-split server is ahead in the measured M=2 and M=3 decode regions, while medium multi-cold-prefill remains a large independent-lane advantage and very-long cold prefill crosses back toward layer split.

This reopens static layer split / Super-Lane as a serious production-topology candidate, but it does not yet change the default contract. Remaining promotion gates are workload mix, unequal prompt/output lengths, session behavior, queue/tail latency, failure isolation, power, and operational flexibility.

Public recipe record: https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe/blob/main/docs/strata-0.1.39-performance-crossover-20261005.md
