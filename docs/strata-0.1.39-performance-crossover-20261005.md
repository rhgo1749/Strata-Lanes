# Strata 0.1.39 performance crossover evidence — 2026-10-05

## Scope and provenance

This record belongs to the validated but not-yet-promoted Strata 0.1.39 sync candidate on `sync/upstream-0.1.39`.

```text
fork branch         sync/upstream-0.1.39
fork candidate      d51d7e9cbc327f90c2fd59b1e20a949033f594e2
upstream Strata     6f32ec070f23ced9f50e704d854d775da52591ab (v0.1.39)
engine              Strata 0.1.39
model/quant         Qwen3.8-Flash-Next GSQ-RCO IQ3_S
host                Ryzen 9 9950X3D / 128 GB DDR5
primary GPUs        RTX 5070 Ti 16 GB x3
driver/CUDA         615.71.09 / CUDA 13.4
context             262144 per independent lane unless noted
resident KV         32768 per independent lane unless noted
sampling            temperature=0, seed=1234
```

The 0.1.39 campaign reused the public recipe workload generator and current server/frontend. The fixed ~1.5K reasoning prompt used for scaling and decode has hash `429fe691f25560a1` and currently tokenizes to 1469 tokens. The retained 0.1.38 recipe recorded 1456 tokens for its earlier frontend path, so cross-version percentage comparisons are contract-matched rather than byte-identical. Within the 0.1.39 fixed-prompt TG challenger comparison, the independent-lane and layer-split arms use the same 1469-token prompt contract.

## Independent-lane scaling

Five retained repetitions per point:

| Active lanes | Common-wall aggregate TG |
| ---: | ---: |
| 1 | **73.63 ± 1.67 tok/s** |
| 2 | **143.19 ± 3.06 tok/s** |
| 3 | **192.16 ± 4.11 tok/s** |

The 3-lane result is 2.610x the one-lane result (87.0% scaling efficiency). Corrected mixed fiction/coding/reasoning serving measured **189.25 ± 5.07 tok/s** over nine retained runs.

## Workload sensitivity

Cold PP remained essentially flat versus the 0.1.38 campaign for the ~15K medium workloads, while warm decode generally improved. Retained 0.1.39 values include:

| Workload | Cold PP | Warm TG |
| --- | ---: | ---: |
| fiction short | 878.6 tok/s | 72.28 tok/s |
| coding short | 840.6 tok/s | 85.70 tok/s |
| reasoning short | 872.2 tok/s | 79.24 tok/s |
| fiction medium | 2799.1 tok/s | 69.22 tok/s |
| coding medium | 2791.5 tok/s | 83.84 tok/s |
| reasoning medium | 2803.4 tok/s | 70.48 tok/s |
| long_review ~110K | 2719.0 tok/s | 66.20 tok/s |

The long-review cold PP mean is **2719.02 ± 1.12 tok/s**. Warm decode at ~110K is slightly below the retained 0.1.38 mean, so this campaign does not claim a universal decode speedup.

## Oversubscription

| Requests | Aggregate TG |
| ---: | ---: |
| 3 | **190.26 ± 3.07 tok/s** |
| 4 | **141.55 ± 0.90 tok/s** |
| 6 | **191.06 ± 2.67 tok/s** |
| 9 | **191.86 ± 1.52 tok/s** |

The partial-second-wave M=4 trough remains. Queue-tail behavior still follows the multi-wave structure rather than turning into an unlimited continuous-batching claim for ordinary lanes.

## Heterogeneous isolation

Retained RTX 5070 Ti + RTX 5060 Ti ABBA:

| Measurement | TG |
| --- | ---: |
| RTX 5070 Ti solo | **75.80 ± 2.21 tok/s** |
| RTX 5070 Ti concurrent | **75.24 ± 1.56 tok/s** |
| RTX 5060 Ti concurrent | **59.98 ± 1.23 tok/s** |
| Common-wall aggregate | **118.68 ± 2.35 tok/s** |

The raw fast-lane concurrent delta is **-0.74%**, again not a material pacing result.

## Shared expert arena PSS

Shared-arena physical memory remains structurally unchanged:

| Context / resident KV | Shared two-engine PSS |
| --- | ---: |
| 32K / 8192 | **52.113 GiB** |
| 262K / 32768 | **58.038 GiB** |

A matched 0.1.39 private two-engine PSS value was not retained: while the second private ~46.8 GiB expert arena was being populated, systemd-oomd killed the DevSpace cgroup under host memory pressure. Do not substitute the 0.1.38 private number as a 0.1.39 measurement.

## Layer-split challenger correction

The first 0.1.39 three-request layer-split probe used the ordinary FIFO server path and therefore measured only a serial control:

- one warm request: **120.62 ± 1.24 tok/s**
- three simultaneous requests, FIFO common-wall aggregate: **118.75 ± 5.98 tok/s**
- short cold PP control: **1047.45 ± 3.20 tok/s**

That FIFO three-request number is **not** the 0.1.39 layer-split concurrency ceiling. Upstream 0.1.39 adds opt-in engine batch slots and layer-split pipeline groups.

### Batch-only control

Three-GPU layer split with `--batch 3` and the shared expert arena, but without pipeline groups, measured:

**132.46 ± 3.61 tok/s** common-wall aggregate TG.

A private-arena repeat measured 133.33 ± 3.41 tok/s, so the shared backing itself did not materially move this TG result.

### Pipelined layer split

The strongest measured challenger used:

```text
GPU order             0,2,1
explicit split        18,34
stages                0-17 / 18-33 / 34-47
--batch               3
--batch-groups        3
--trim-stage-weights
--shared-expert-arena enabled
```

The engine log confirmed `strata batch (pipelined, 3 groups of 1)`. Three fixed 512-token reasoning requests measured:

**209.66 ± 4.02 tok/s** common-wall aggregate TG.

Against the matched 0.1.39 independent-lane result of **192.16 ± 4.11 tok/s**, the pipelined layer-split challenger is **9.1% higher** in this fixed three-request decode region. This supersedes any interpretation that the FIFO control represented the best 0.1.39 layer-split concurrency path.

The gain is coupled to pipeline execution and stage trimming: `--batch 3` without groups/trim reaches only **132.46 tok/s**, while the pipelined arm reaches **209.66 tok/s**.

## Cold-prefill crossover

To test prompt processing rather than warmed prefix reuse, every retained request used a unique nonce and reported `cache_n=0`. Three cold prompts were submitted simultaneously. The common-wall PP metric is total newly processed prompt tokens divided by the time until all three requests completed their one-token response.

| Cold prompt regime | Independent lanes 1+1+1 | Pipelined layer split | Relative result |
| --- | ---: | ---: | --- |
| ~15K x3 | **5901.34 ± 55.16 tok/s** | 3289.19 ± 18.83 tok/s | **Lanes +79.4%** |
| ~110K x3 | 5822.71 ± 4.49 tok/s | **6028.09 ± 14.50 tok/s** | **Layer split +3.5%** |

The ~15K layer-split requests finish in a near-serial admission staircase (~4.6 s, ~9.1 s, ~13.7 s), even though each individual prompt runs at roughly 3.3K PP. Independent lanes instead prefill all three prompts concurrently.

At ~110K, the layer-split intra-prompt chunk pipeline is fully utilized: each prompt processes at roughly 6.08-6.12K PP and the three-request common wall is ~54.7 s. That is enough to slightly exceed the independent-lane common-wall aggregate despite serial request admission.

The 15K and 110K A/Bs are cold/no-reuse and workload/length matched, but the per-arm nonce text is not byte-identical (the harness encoded the arm name in the nonce). The crossover is large enough to be operationally interesting, but exact percentage claims should retain this caveat.

## Architecture interpretation

Strata 0.1.39 changes the static challenger landscape:

- independent lanes remain strongly favored for multiple medium-length cold prompts and preserve separate failure/session domains;
- upstream pipelined layer split can beat independent lanes for matched three-request decode on this host;
- sufficiently long prompts can also cross over in favor of one pipelined layer-split engine because intra-prompt chunk pipelining becomes highly efficient;
- the ordinary FIFO layer-split result is only a control and must not be used as the 0.1.39 concurrency ceiling.

This does **not** by itself promote a Super-Lane or replace the production independent-lane topology. Real serving mixes include unequal prompt/output lengths, session affinity, queueing, failures and topology reconfiguration costs. The correct next architecture question is workload-aware topology selection, not a universal winner claim.
