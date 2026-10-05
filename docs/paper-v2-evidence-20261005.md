# Paper v2 evidence snapshot — 2026-10-05

**Evidence date:** 2026-10-05 (KST)

## Status

- **Only paper v1 has been submitted.**
- No paper v2 manuscript or revision has been submitted as of this snapshot.
- `paper-v2-evidence` is a post-v1 evidence anchor for a possible future revision.
- This tag does **not** mean that a v2 manuscript already exists or has been submitted.
- Reserve `paper-v2` for an actual revised manuscript/submission state.
- The earlier `docs/paper-v2-evidence-20261001.md` remains a historical snapshot and is not rewritten.

## Repository / engine provenance

### Strata-Lanes

- Repository: `rhgo1749/Strata-Lanes`
- Promoted software generation: **Strata 0.1.39**
- Upstream Strata: `6f32ec070f23ced9f50e704d854d775da52591ab`
- PR #24 software merge to main: `9ae0839b9f376dca9742804924db05902f872ad8`
- Promotion-documentation commit: `aaf843090d9a8239ca91fb9a40adfa337a80ddea`
- Final validated sync head before merge: `1a454ca80d698af863693c1b2d3fafac0b3b1ceb`
- Performance binary source generation: `d51d7e9cbc327f90c2fd59b1e20a949033f594e2`
- Measured binary SHA-256: `9aa71607ca3c322c61e75e2fbbb0cbd8f1816f663c9c856b6df342ecd9e18c05`

The later commits after the measured binary are documentation/promotion-state changes; the retained performance data stays attributed to the source generation that produced it.

### Reproducibility recipe / retained evidence

- Repository: `rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe`
- Current evidence/promotion main: `3fb35129345615790495b13d1a573a9f8e27ecf2`
- Raw retained evidence: `bench/raw/0.1.39-20261005/`
- Compact topology table: `bench/layer-split-ab-0.1.39-20261005.csv`
- Human-readable topology record: `docs/strata-0.1.39-performance-crossover-20261005.md`

## Hardware / model provenance

- CPU: Ryzen 9 9950X3D
- RAM: 128 GB DDR5
- Primary GPUs: RTX 5070 Ti 16 GB ×3
- CUDA: 13.4
- NVIDIA driver: 615.71.09
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S
- Layer-split GPU order: `0,2,1`
- Explicit split: `18,34`
- Pipeline challenger: `--batch 3 --batch-groups 3 --trim-stage-weights` with shared expert arena

## Current topology evidence

### Decode concurrency

| Concurrent requests | Independent lanes | Three-GPU pipelined layer split | Result |
|---:|---:|---:|---:|
| M=1 | 73.63 ± 1.67 tok/s | **120.62 ± 1.24 tok/s**¹ | layer split +63.8% |
| M=2 | 143.19 ± 3.06 tok/s | **147.84 ± 2.26 tok/s** | layer split +3.25% |
| M=3 | 192.16 ± 4.11 tok/s | **209.66 ± 4.02 tok/s** | layer split +9.11% |

¹ M=1 is the retained same-binary three-GPU layer-split single-request control; it was not rerun under the exact batch3/groups3/trim configuration. M=2 and M=3 are exact measurements of the fixed pipelined configuration.

For M=2, both requests completed together at roughly 6.9 s mean E2E. The engine remained on the pipelined `3 groups of 1` path with one group unused. This closes the previously missing M=2 point.

### Cold-prefill crossover

Every retained PP request used a unique nonce and reported `cache_n=0`.

| Cold prompt regime | Independent lanes | Pipelined layer split | Result |
|---|---:|---:|---:|
| ~15K ×3 | **5901.34 ± 55.16 tok/s** | 3289.19 ± 18.83 | lanes +79.4% |
| ~110K ×3 | 5822.71 ± 4.49 | **6028.09 ± 14.50** | layer split +3.5% |

The 15K/110K arms are workload/length matched but not byte-identical because the nonce text encodes the topology arm.

## What changed from the 2026-10-01 evidence snapshot

The old paper-v2 evidence snapshot treated ordinary three-request layer split as a FIFO/serial challenger and therefore concluded that independent lanes dominated concurrent serving.

That interpretation is **superseded for Strata 0.1.39**.

Upstream 0.1.39 adds batch slots, layer-split pipeline groups and stage-weight trimming. The ordinary FIFO result is now retained only as a control. With the explicit three-stage pipeline, layer split wins the measured fixed decode region at M=2 and M=3, while independent lanes still have a large advantage for medium-length simultaneous cold-prefill and the very-long PP region crosses back toward layer split.

The revision-safe conclusion is therefore a **workload-dependent topology crossover**, not “lanes always win concurrency” and not “layer split always wins.”

## Intended future revision use

If a paper revision is requested, this evidence can support:

1. A same-generation Strata 0.1.39 comparison between independent request-level lanes and the upstream-native pipelined layer-split execution path.
2. A decode-concurrency result covering M=1/M=2/M=3 rather than only single-request and three-request endpoints.
3. A prompt-length crossover result showing that request-level independent prefill dominates at ~15K ×3 while intra-prompt layer pipelining catches and slightly passes it at ~110K ×3.
4. A corrected interpretation of the old FIFO layer-split control.
5. The continuing shared-arena / independent-lane systems evidence as historical and supporting material, without relabeling measurements from older Strata generations.

## Naming convention

- `paper-v1`: submitted manuscript state.
- `paper-v2-evidence`: current post-v1 experimental evidence anchor; keep it aligned with the repository main snapshot that contains this manifest.
- `paper-v2`: reserve for an actual revised manuscript/submission state.
