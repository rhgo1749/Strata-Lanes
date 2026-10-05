# Multi-GPU runtime roadmap

This roadmap covers the experimental multi-GPU serving runtime preserved in this research fork. The repository no longer presents independent lanes as the recommended general-purpose deployment; upstream Strata is the default user-facing path. The independent-lane design remains a reference experimental baseline for controlled comparisons.

Roadmap authority is GitHub Issue #1 and its child Issues. This document summarizes the durable direction. Concrete reference-host hardware, tuning values, benchmark tables, and production validation records live in the separate public recipe repository: [`rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe`](https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe).

## Current reference experimental baseline

The current promoted engine generation is **Strata 0.1.39**, integrated from upstream `6f32ec070f23ced9f50e704d854d775da52591ab` through PR #24 and promoted on main at `9ae0839b9f376dca9742804924db05902f872ad8`. The retained reference topology remains the independent-lane architecture while consuming upstream batching, Responses API, conversation-cache telemetry, decode/kernel, CPU-pool, expert-cache, low-RAM/file-tier, and layer-split improvements. Ordinary lanes strip inherited upstream internal batching just as they already strip inherited layer-split/peer execution, so those mechanisms remain explicit challenger/backend primitives rather than accidental replacements for request-level lanes.

The architectural baseline is:

- one ordinary Strata engine process per GPU lane;
- one upstream-native shared host expert arena;
- lane-local CUDA state, hot-expert cache, resident KV, session state, and failure boundary;
- per-lane context / resident-KV budgets;
- capability-aware routing such as vision-lane constraints;
- session affinity plus hardware-agnostic live-state-aware placement;
- routing above the engines rather than token-, layer-, or expert-level GPU synchronization.

Matched validation keeps the intended workload split: upstream layer-split remains a useful single-request challenger, while independent lanes remain a reference baseline for concurrent-serving experiments. Hardware-specific measurements and caveats belong in the recipe repository.

## Design rule

Prefer coarse-grained request/session parallelism while it wins on the workload that matters.

More sophisticated mechanisms are not automatically better. Cross-GPU expert routing, migration, dynamic shared KV, learned control, or single-process distributed execution add synchronization, coupling, and larger failure domains. Introduce them only after the simpler serving-control stages below leave a measured gap.

## Phase 1 — Benchmark, observability, and interference characterization (completed)

Before changing policy, make the real decision variables observable.

Measure at least:

- queue delay, TTFT, E2E latency, TPOT/ITL where available, and throughput/goodput;
- reusable prefix / new-prefill work;
- active and queued work;
- session turn / stable session identity where available;
- live prompt/KV footprint;
- hot-expert hit/miss behavior;
- host RAM, CPU, PCIe/interconnect, GPU/VRAM, and power where measurable;
- lane health/restart and hard capabilities such as vision.

The workload matrix must separate cold/no-reuse, warm-prefix, multi-turn continuation, long-context, heterogeneous lengths, capability-constrained routing, `M > N` overload, cancellation/failure, and representative agent workloads.

### Matched interference probes

Run the same target request/lane under solo and concurrent conditions. Vary the other lanes' work while holding the target workload fixed.

The goal is to determine whether target-lane service cost is adequately explained by local work/load or whether a repeatable residual tracks shared host-memory / PCIe / expert-arena pressure.

Do not introduce a coupled cost model merely because resources are shared.

**Exit condition:** the baseline can be replayed, routing decisions are auditable, and shared-resource interaction is either shown immaterial or characterized well enough to test as a scheduler signal.

## Phase 2 — Stateful serving control while lanes remain independent (completed)

Phase 2 was evidence-gated and ordered. The promoted new-session placement policy is `balanced-additive-new-prefill-retained-state-proxy-v1`; the retained shared-pressure placement coefficient, bounded-admission default for the all-complete-immediately workload, and workload-regime adaptation were all evaluated and not promoted. `safe-affinity-live-state-v1` remains the rollback control.

### 2A — Strong simple placement baselines

Compare:

- first-free / round-robin;
- least-loaded;
- current session-affinity + live-state control;
- cache-aware + imbalance fallback;
- additive new-prefill + load cost;
- multiplicative new-prefill × load cost;
- session-first balance + cache-aware continuation.

Prefer the simplest policy that captures most of the gain. Benchmark-only online challengers keep existing-session affinity and all health/capability/FIFO constraints intact; they may change only new-session placement among otherwise eligible idle lanes. Because an idle candidate has no active decode load under the current one-request-per-lane contract, retained-state heuristics must be named and reported as proxies rather than mislabeled as engine-truth least-loaded scheduling.

### 2B — Coupling-aware cost only if Phase 1 proves it is useful

If matched interference experiments leave a reproducible residual, add the smallest observable shared-pressure term that improves held-out prediction and end-to-end serving.

Validate on unseen workload combinations, especially high-demand/high-pressure cases. Keep resource telemetry as the mechanism evidence.

### 2C — Admission and tail control

Placement is insufficient when the system is overloaded.

Under `M > N`, evaluate a bounded decision:

```text
route now | wait for a better lane | defer
```

Measure p50/p95/p99 queue delay and TTFT, E2E latency, goodput/TPS, starvation/fairness, active-session/cache pressure, and cancellation while queued/deferred.

The objective is to avoid throughput wins that hide tail collapse or sustained cache/state thrashing.

### 2D — Workload-regime adaptation only if necessary

Test the selected fixed policy across session-heavy, short-request, bursty, long-context-heavy, capability-mixed, and heterogeneous-length workloads.

If one fixed policy remains robust, stop.

If it degrades materially, adapt only a small set of interpretable policy weights/thresholds from recent telemetry. Learned/RL control is not justified unless simpler feedback leaves a measured gap.

**Exit condition:** the serving control improves a declared end-to-end or tail objective over strong simple baselines without regressing correctness, fairness, or failure isolation.

## Phase 3 — Startup/runtime lifecycle overhead (implemented / validated)

The promoted design removes repeated shared-arena source loading without changing the independent-lane inference model.

Implemented contract:

- lane 0 is the authoritative population leader for each supervisor generation;
- the shared header is marked incomplete before population and ready only after the full source load succeeds;
- later sequential lanes attach as followers, verify size/pack identity/readiness, and skip the repeated source expert load;
- the supervisor holds one non-blocking ownership lock for the arena pathname, preventing concurrent supervisors from repopulating the same backing;
- a new leader repopulates an existing compatible backing rather than trusting stale contents as persistent cache state;
- default supervisor-managed tmpfs backing is removed on graceful exit; explicit `--arena-file` lifetime remains operator-owned.

The reference-host promotion gate showed a matched 3-lane startup reduction while preserving text, vision, multi-turn affinity, malformed-input, cancellation, and recovery behavior. Hardware-specific timing belongs in the recipe repository.

Automatic child-process respawn and persistent cross-run arena reuse are not introduced by this phase. They remain separate lifecycle work only if a measured operational need justifies them.

## Conditional architecture challengers

These are evidence-triggered branches, not mandatory phases.

### Decode-assist helper GPU (deferred on 0.1.38; reopen on new evidence)

Evaluate upstream Strata's P2P-free secondary expert caches before adding more topology coupling. This challenger keeps one ordinary lane as the owner of dense/attention execution, KV, session state, MTP/draft state, local expert cache, API semantics, and failure recovery. A secondary GPU is used only as an expert worker for decode/MTP through upstream `--expert-cache-device1/2/3` and pinned host staging. Prompt prefill remains on the primary GPU.

This is deliberately distinct from both layer split and the peer tier:

- unlike layer split, it does not partition the model's layer graph or require prompt chunks/tokens to traverse multiple GPUs;
- unlike `--peer-device`, it does not require a validated P2P path;
- unlike cross-lane migration, it does not move KV or session ownership;
- if the helper is absent or not worthwhile, the request remains an ordinary independent-lane request.

The first reference-host experiment is tracked by [#23 — P2P-free decode-assist helper GPU](https://github.com/rhgo1749/Strata-Lanes/issues/23). Test the RTX 5060 Ti as the first helper because it can preserve the 3×RTX 5070 Ti production lane pool, then test a RTX 5070 Ti helper only as an opportunity-cost reference. Use explicit disjoint CPU affinity for all participating processes; a helper process without CPU partitioning can create a false cross-lane slowdown.

Measure the primary-only control against primary+helper with fixed prompt/model/context/KV conditions. Separate cold/no-reuse prompt processing from sufficiently long warm decode, record speculative acceptance, useful secondary-expert work, CPU-pool drain, cache behavior, PCIe traffic where available, power, and failure fallback. Also account for the helper GPU's opportunity cost as an independent lane.

Promotion requires a repeatable end-to-end decode/E2E improvement without a material prompt-path regression, without meaningful degradation of unrelated lanes through host/PCIe contention, and with a clean primary-only fallback. Prefer this lower-coupling mechanism over a Super-Lane when it closes the same single-request gap.

**Current status on Strata 0.1.38 / the reference host: deferred, not rejected.** The retained helper sweeps show real CPU expert-pool work reduction but no established end-to-end decode gain because the primary-GPU stage remains the critical path. Do not spend more reference-host effort on slot tuning under the same execution contract. Reopen with a short matched smoke before any full campaign when upstream materially changes remote-expert transport/overlap/synchronization, when primary-GPU execution becomes faster enough to expose CPU expert drain, or when testing a materially more CPU-constrained host/workload.

### Super-Lane execution groups

Treat a Super-Lane as a scheduler-level execution group: one logical lane may own either one GPU or a fixed multi-GPU Strata engine while the rest of the serving control plane continues to route whole requests/sessions between logical lanes.

Treat static layer split as the higher-coupling challenger after decode assist. On a three-GPU host, compare the production `1+1+1` independent-lane baseline against a `2+1` topology: one two-GPU Super-Lane plus one ordinary lane. Use upstream Strata execution primitives rather than forking model execution:

- use upstream layer split as the static backend where P2P is unavailable or unnecessary;
- keep upstream peer-device execution only for hardware where the required peer path is validated;
- do not conflate P2P-free `--expert-cache-device1/2/3` decode assist with a Super-Lane; it remains the lower-coupling challenger above;
- keep session affinity, capabilities, admission, health checks, telemetry, shared-arena ownership, and failure reporting at the logical-lane boundary.

Measure the crossover rather than assuming aggregation wins: single-request TTFT/E2E and prompt/decode throughput, concurrent aggregate goodput, queue/tail latency, context capacity, shared host-memory/PCIe pressure, power, and failure-domain cost.

Do **not** begin with dynamic GPU bonding. Reopen bond/unbond policy only if the static `1+1+1` versus `2+1` experiment demonstrates a repeatable workload-dependent crossover large enough to justify topology changes. Any dynamic version must retain a clean independent-lane fallback and must not migrate an active session merely to rebalance GPUs.

**Current status on the promoted Strata 0.1.39 engine / reference host: reopened as a serious default-topology challenger; the layer-split topology itself is not yet promoted.** Upstream 0.1.39 materially changed the layer-split server through batch slots, pipeline groups and stage-weight trimming. The fixed three-GPU `18,34` pipeline with `--batch 3 --batch-groups 3 --trim-stage-weights` and the shared expert arena now beats independent lanes in the measured fixed decode region at both M=2 (**147.84 ± 2.26 vs 143.19 ± 3.06 tok/s, +3.25%**) and M=3 (**209.66 ± 4.02 vs 192.16 ± 4.11, +9.11%**). The retained M=1 layer-split control is also much faster than one lane, although that point was not rerun with the exact pipeline config. Cold-prefill is the important counterexample: ~15K x3 strongly favors lanes (**5901.34 vs 3289.19 tok/s**), while ~110K x3 narrowly favors the pipelined split (**6028.09 vs 5822.71 tok/s**). The next gate is therefore no longer more fixed microbenchmarks; it is mixed prompt/output lengths, session behavior, queue/tail latency, power, failure isolation and operational flexibility under the always-on three-GPU layer-split topology. Full evidence: [`strata-0.1.39-performance-crossover-20261005.md`](strata-0.1.39-performance-crossover-20261005.md).

### Elastic Super-Lane lifecycle

Treat upstream engine load/unload and lazy-start controls as lifecycle primitives, not as a reason to introduce dynamic GPU bonding early. **This challenger remains deferred rather than closed.** Evaluate it only after a future static Super-Lane recheck demonstrates a repeatable workload-dependent advantage large enough to survive drain/reconfigure/load/restore cost.

The first elastic experiment should use an explicit drain/reconfigure/restore sequence:

- stop admitting new work to the donor lane and wait for its active request to finish;
- unload or stop that lane without discarding another lane's active session;
- form the pre-declared Super-Lane topology from the released GPU;
- run the target workload and record reconfiguration, model-load, queue, TTFT/E2E, throughput, power, and failure-recovery cost;
- dismantle the Super-Lane and restore the donor ordinary lane before returning it to admission.

Prefer upstream `lazy_load`, `/v1/load`, `/v1/unload`, `idle_unload_s`, `min_free_vram_mib`, and related server lifecycle primitives over adding a fork-specific engine lifecycle.

Do not migrate an active session merely to free a GPU, and do not promote automatic bond/unbond policy unless the static crossover remains large enough after reconfiguration and reload overhead are included. The ordinary independent-lane topology remains the rollback path.

### Single-process multi-GPU execution

Prototype only if single-request underutilization is a material target bottleneck. Compare single-request latency/throughput, concurrent aggregate throughput, synchronization/interconnect cost, context capacity, power, and failure-domain cost.

### Distributed/coordinated hot-expert cache

Prototype only if host expert misses/traffic remain a dominant steady-state cost after scheduling improvements. Promotion requires reduced misses to outweigh new GPU-to-GPU communication and coordination.

### Profile-guided expert-cache and lane routing

Use upstream learned expert profiles only as an opt-in serving signal. First make profile ownership explicit: ordinary independent lanes must not write the same `--expert-profile-save FILE`; use lane-local profile files or another single-writer aggregation contract before enabling persistence.

Stage the challenger:

- first, measure whether independently learned per-lane expert profiles remain materially different under repeated workload classes and whether those differences improve cache hit rate, H2D traffic, TTFT/E2E latency, or throughput versus one shared/static profile;
- only if a stable, useful difference exists, expose profile/cache affinity to the scheduler as one bounded new-session placement signal alongside existing health, capability, session-affinity, and load constraints;
- consider supervisor-side aggregation into a global profile only if it outperforms lane-local profiles or reduces operational complexity without erasing useful specialization.

Do not infer request semantics from profile identity alone and do not introduce learned routing merely because `--expert-profile-save` exists. Promotion requires a held-out end-to-end serving gain after controlling for workload mix, cache warmth, and expert-profile initialization, with a clean fallback to the current profile-agnostic scheduler.

### Lane-local conversation parking

Evaluate upstream conversation snapshots as a bounded multi-session capability **inside one ordinary lane** before attempting cross-lane migration. Keep the production default at `--conversation-cache-mib 0` until the scheduler explicitly understands parked-state ownership and the experiment passes its gate.

The first challenger should preserve lane ownership: a session may be parked to host RAM and later restored only by the same lane. Compare the current one-live-session-per-lane baseline against bounded parking under alternating and overloaded `M > N` workloads. Measure host-RAM cost, snapshot save/restore latency, reusable KV/prompt work, queue delay, TTFT/E2E, aggregate goodput, eviction behavior, cancellation/failure recovery, and the cost of falling back to ordinary prompt processing after a miss or eviction.

Do not treat upstream's internal snapshot LRU as an independent placement policy. The Lanes supervisor must remain the authority for session/lane affinity and admission; parking is a lane-local retained-state mechanism underneath that contract. Start with explicit RAM/slot limits and no cross-lane snapshot transfer.

Promotion requires a repeatable end-to-end advantage over wait/recompute with bounded RAM and no regression in affinity, fairness, correctness, or failure isolation.

**Current mechanism status on Strata 0.1.38 / the reference host: passed, but not production-enabled.** A direct ordinary-lane synthetic alternating-session probe showed that a sufficiently sized same-lane cache materially reduces return-turn E2E: with six conversations, six parking slots reduced mean second-turn E2E from about 3.64 s to 2.81 s (~22.7%), with restores averaging about 11.9 ms and the parked cache growing to about 2.0 GiB. The same six-conversation workload with only four slots reached seven evictions and erased the gain, demonstrating that bounded capacity/working-set fit is part of the serving contract rather than a tuning detail.

The Lanes supervisor already remembers multiple stable affinity keys per lane and sends returning sessions back to their remembered lane. A production-shaped **three-lane supervisor A/B has now passed the performance/control-plane portion of this gate** while keeping strict same-lane affinity and ordinary recompute on a cache miss. With 4096 MiB / 4 slots per lane, cold/new-session turn 1 remained essentially unchanged; on returning turn 2, M=6 mean E2E fell about 37.8% and aggregate completion TPS rose about 54.3%, while M=9 mean E2E fell about 33.6% and aggregate completion TPS rose about 61.5%. Queue time fell in the same direction. The sequential 4→6→9 campaign also produced 1–2 engine-side evictions per lane, which safely recomputed but confirms that the supervisor must observe actual parked occupancy rather than equating affinity ownership with snapshot residency.

**Robustness/observability status: passed on the reference host; production default still unchanged.** Live engine `DONE` telemetry for parked count/bytes/evictions is surfaced through per-lane supervisor status. Client disconnect/cancellation cleans up correctly, forced slot pressure produces engine-truth evictions followed by successful same-lane `cached_tokens=0` recomputation, and host cache/memory admission tests pass. The prior child-engine restart blocker is resolved by separating **new-session eligibility** from **affinity ownership lifetime**: new sessions still require a loaded engine, while an existing affinity is retained as long as its private lane wrapper lives. If that wrapper is healthy but its Strata child died, the returning affinity is allowed to reach the same wrapper and trigger `server.py`'s synchronous child restart path. In live validation, an unrelated new session avoided the dead lane, the existing affinity remained owned by lane2, and one returning request restarted the child and completed HTTP 200 on lane2 in about 5.1 s with `cached_tokens=0` recomputation. The controlled production-shaped canary has passed, and a subsequent **reference-host deployment matched soak** reproduced the benefit with the same patched binary in both arms. The deployment path was held constant around the standard Lanes supervisor. Under six mixed agent-like stable sessions, cold/wake turn 1 was unchanged (~0.1%), while parking reduced returning-turn wall time by 35.7% / 32.2%, reduced mean E2E by 28.5% / 25.7%, and raised aggregate completion throughput by 55.4% / 59.1% on turns 2 / 3. The ON arm ended with two affinities per lane, one parked conversation per lane, ~1.60 GiB total engine-reported parked state, zero evictions and zero queue depth. The deployment gate was followed by a **Hermes `eval` live-use validation** using real Hermes session persistence and the Strata provider against the standard Lanes supervisor through the same reference-host deployment path. Six named Hermes sessions were alternated and resumed repeatedly. Their prompts were roughly 25K tokens; normal returning requests reused about **25.1K–25.6K tokens** while reading only ~232–264 fresh tokens. With the reference **4096 MiB / 4 slots / 8192 MiB floor per lane** contract, lane1 eventually produced one real byte-budget-driven eviction; the affected revisit safely fell back to partial reuse (`16384` reused, ~9.5K freshly read) with conversation continuity intact. Final observed parked state was approximately **1.73 / 2.86 / 1.10 GiB** on lanes 0/1/2 with eviction counts **0 / 1 / 0**, zero queue depth, and no failed requests. On the reference host the capability is now **enabled in production** on the three RTX 5070 Ti lanes; the RTX 5060 Ti remains outside the serving pool. Keep rollback to `--conversation-cache-mib 0` immediate if RAM pressure, recovery errors, or latency regressions appear. Cross-lane snapshot migration remains deferred.

### Dynamic/shared KV or migration

Keep cross-lane migration deferred until lane-local conversation parking has been evaluated and every lane can still admit the required context with placement/wait/recompute as a clean fallback. Reopen only when measured capacity/utilization or overload behavior shows that same-lane parking is insufficient and the ownership, transfer, and recovery complexity is justified.

Research tracker: [#20 — cross-lane parked conversation migration](https://github.com/rhgo1749/Strata-Lanes/issues/20). Keep implementation parked while upstream Strata's conversation snapshot/cache/storage-tier interfaces are evolving; prefer consuming upstream state primitives over forking their snapshot format.

## Promotion gate

A challenger must demonstrate all of the following on repeated runs:

1. material improvement on a declared real workload objective;
2. no regression in required context/admission behavior;
3. no regression in tool/API, streaming, cancellation, malformed-input, and failure behavior;
4. stable memory use and lane health under soak;
5. an explainable gain after cold/warm/cache state is controlled;
6. acceptable power and interconnect cost;
7. held-out workload/regime validation where policy fitting is involved;
8. a clean independent-lane fallback.

## Boundary with model-internal research

This roadmap is **serving/runtime only**. Model-internal training or architecture research is out of scope for this repository and is not a prerequisite for serving progress.

## Explicit non-goals

Until measurements justify them, this roadmap does **not** assume that Strata should:

- become a general tensor-parallel or pipeline-parallel engine;
- require GPU-to-GPU expert traffic for normal serving;
- merge lane-local session state into one distributed failure domain;
- implement unified KV merely for symmetry;
- introduce learned scheduling before simpler controls fail;
- make model-internal research a prerequisite for serving progress;
- optimize benchmark aesthetics at the expense of real serving behavior.

## Near-term order

1. Keep the completed Phase 1/2 serving-control and Phase 3 lifecycle gates as regression controls on the promoted **0.1.39** software baseline; retain measured 0.1.30/0.1.31/0.1.38 evidence under its original engine generation.
2. Do not add another mandatory serving phase without a measured residual.
3. Keep decode assist as **0.1.38 evidence-negative / deferred**, but treat static layer split / Super-Lane as **reopened on 0.1.39** because the measured batch-group pipeline now crosses independent lanes in some decode and very-long-prefill regions. Do not collapse the result into a universal winner claim.
4. Before any topology promotion, test mixed prompt/output lengths, queue/tail behavior, power, failure-domain cost and the drain/reconfigure/restore lifecycle against the independent-lane fallback. Elastic Super-Lane remains gated on this broader static crossover holding after reconfiguration cost; cross-lane migration stays later.
5. Treat upstream helper, peer, layer-split and load/unload controls as distinct execution/lifecycle primitives: helper offload is valid without P2P, peer execution is deferred on hardware without a validated P2P path, and neither automatically replaces independent lanes.

The default bias remains deliberate simplicity: add coupling only when measurements show it buys something.
