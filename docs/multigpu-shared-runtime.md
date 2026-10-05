# Experimental multi-GPU shared runtime

This fork provides an opt-in Linux runtime that keeps one ordinary Strata engine process per GPU lane while sharing the large host expert arena between processes.

Detailed reference-host hardware, tuning values, benchmark tables, and validation records live in the separate public recipe repository: [`rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe`](https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe).

## Runtime contract

- One Strata engine process runs per GPU lane.
- The large host expert arena is backed by a shared Linux mapping when the opt-in shared-arena settings match the exact expert allocation size.
- CUDA state, hot-expert cache, resident KV, session state, and process lifecycle remain lane-local.
- Host-KV capacity is configured per lane rather than through a dynamic cross-lane allocator.
- CPU affinity is partitioned between lanes by default so independent expert worker pools do not collide on the same physical cores.
- Generation requests are leased to one lane for the duration of the request or stream.
- Conversation requests preserve lane affinity when the supervisor can identify a stable session, so lane-local KV/checkpoints survive across turns instead of being discarded by request-level round-robin routing.
- The default single-GPU Strata path is unchanged when the shared-arena mode is not enabled.

## Shared arena

The promoted Strata 0.1.39 baseline retains the file-backed arena primitive introduced upstream in 0.1.30 through `--shared-expert-arena FILE` (originating from this fork's upstream PR #129). On Linux, upstream `PinnedArena` maps one shared file with a 4 KiB header containing the arena geometry and pack fingerprint; incompatible size or pack identity is refused before the arena is used. The ordinary anonymous/hugetlb path remains unchanged when the option is absent. The Linux whole-arena pinning behavior used by the lane runtime remains intact; the WDDM shared-memory cap is Windows-specific unless explicitly overridden with `STRATA_ARENA_PIN_GIB`.

The lane supervisor composes that upstream primitive rather than intercepting `mmap`: it strips inherited shared-arena options, adds the same explicit `--shared-expert-arena` path to every shared lane, and removes the option entirely for the private-arena benchmark arm. The default backing lives under `/dev/shm`, matching upstream's tmpfs contract; an explicit `--arena-file` selects an operator-managed path. Automatically chosen backing files are removed on graceful supervisor exit, while explicit backing-file lifetime remains operator-owned.

Phase 3 adds a leader/follower population contract without changing the steady-state inference model. Sequential startup makes lane 0 the authoritative population leader. Before copying experts it publishes the shared header as incomplete; only after the full expert source load succeeds does it publish readiness with release semantics. Later lanes start with `--shared-expert-arena-follower`, verify the existing size/pack identity and readiness state, and then skip the repeated source expert load. A follower refuses an incomplete arena rather than observing partially populated weights. The supervisor also holds a non-blocking ownership lock for the arena pathname for its lifetime, so a second supervisor cannot repopulate the same backing concurrently. A new leader intentionally repopulates an existing compatible backing rather than trusting stale bytes as persistent cache state.

## Lane-local state

Every lane keeps its own normally resident GPU weights, GPU hot-expert tier, CUDA streams and graphs, GPU-resident KV window, conversation/session state, and failure boundary. The authoritative long-context KV continues to use Strata's existing host-memory streaming path.

## CPU and topology tuning

The supervisor can automatically partition physical CPU cores between lanes. It also exposes optional per-lane CPU, PCIe/cache, context, and resident-KV controls for asymmetric hosts.

Those values are intentionally not prescribed here. They are hardware-specific and should be measured on the target system; the public recipe repository contains one concrete reference-host example.

## Hardware guidance

The multi-GPU path is not tied to one GPU model or lane count. The practical rule is that **every selected GPU must first be able to run one usable single-GPU Strata lane**, while the host must have enough RAM, CPU and PCIe capacity for all lanes concurrently.

A concise starting guide:

- shared-arena multi-process mode: Linux;
- GPU count: 2 or more NVIDIA GPUs;
- VRAM: satisfy the chosen single-GPU Strata configuration on every lane; 16 GB+ per GPU is a useful multi-lane target for additional hot-cache/KV headroom;
- RAM: one shared expert arena + every lane's host-KV + OS/runtime headroom;
- CPU: the current automatic partitioner needs at least 2 physical cores per lane; 4–6 physical cores per active lane is a more practical starting target when available;
- PCIe: confirm the negotiated link for every card and measure asymmetric lanes rather than copying another host's `pcie-frac` values;
- storage: SSD, preferably NVMe;
- NVLink: not required by the normal GPU-per-lane decode path.

The validated 3-lane IQ3_XXS reference configuration uses 128 GB RAM, three 16 GB GPUs, 262K context per lane and a 16-core CPU. Those values are a **validated reference**, not universal minimum requirements.

See [`multigpu-hardware-guide.md`](multigpu-hardware-guide.md) for the sizing rationale, RAM model, CPU/PCIe guidance and bring-up checklist.

## Host-KV capacity

The current implementation treats host-KV as per-lane capacity, not as a dynamic cross-lane pool. Each engine process owns an independent host-KV budget and requests are scheduled to one free lane at a time.

Operators can choose equal or asymmetric context limits according to system RAM and workload requirements. If every lane can expose the required full model window, a dynamic cross-lane KV allocator may provide little practical benefit relative to the coordination complexity it adds.

GPU-resident KV is also per lane. Increasing it can displace the GPU hot-expert cache, so it should be tuned together with expert residency rather than maximized in isolation.

## Request-level parallelism

The production parallelism unit is a whole request or session, not a token, tensor, layer, or expert. This avoids mandatory cross-GPU communication in the normal decode path and preserves comparatively small failure domains.

The supervisor prefers explicit `X-Strata-Session-Id`, conversation/session/thread identifiers in the request, and otherwise derives a privacy-safe best-effort key from the first user message. It keeps a bounded LRU mapping from session keys to lane indices, so several conversations may remain associated with the same engine and use Strata's per-engine prompt-cache checkpoints. A later turn waits for its remembered lane when that lane is busy rather than spilling to another GPU and forcing a full prompt reread; while it waits, that lane is reserved from newly awakened sessions so condition-variable wake-up order cannot steal the cache-rich lane. New sessions use a compatible FIFO wait queue: capability-constrained work such as vision does not block unrelated lanes, but otherwise older compatible waiters receive a newly released lane first. Among eligible idle lanes, the promoted placement policy first prefers the lane with the fewest remembered affinity sessions, then minimizes estimated new-prefill bytes plus the retained last-request byte proxy, then uses live-state recency and rotation as tie-breakers. This prevents long-lived retained-state/cache advantages from repeatedly attracting every new session to one lane while preserving locality among equally balanced candidates. The rollback-safe `safe-affinity-live-state-v1` policy remains available explicitly. Both policies are lane/GPU agnostic: no GPU index, model, or PCIe-width preference is hard-coded. Using a lane for another conversation does not erase older session-to-lane mappings. Requests with no derivable session key still participate in the same busy-lane exclusion and compatible FIFO queue; without a stable key they do not contribute persistent cross-turn affinity.

The trade-off is explicit: preserving a live session can leave another GPU idle briefly or add queueing behind that session's lane. That is preferable to repeatedly paying long-context prefill for the same conversation. More tightly coupled multi-GPU designs remain roadmap challengers and must demonstrate an end-to-end win before promotion.

For controlled serving measurements, `--bench-trace-jsonl FILE` is an opt-in observability path and does not alter the ordinary serving policy. Trace schema 2 writes a `benchmark_manifest` at supervisor startup with the fork commit when available, config hashes, lane topology/capabilities, context/KV/CPU/PCIe settings and shared-arena identity. Each generation request then writes a `lane_lease` record with queue-entry, admission, lane-start, first-response-byte and release timestamps; exact scheduler queue wait, service and end-to-end time; HTTP/completion/error outcome; and response byte count. For streaming requests, first response body byte is also reported as supervisor-observed TTFT; non-streaming first-byte time is retained separately and is not mislabeled as token TTFT.

The same lease record contains an auditable scheduler snapshot: active/queued work, session turn when a stable identity exists, the selected reason, and per-lane health/capability/live-state/placement-key components. Phase-2 instrumentation distinguishes the lane's retained post-request state (`live_request_bytes`, retained prompt-message bytes and sequence) from work that is actively executing: each busy lane exposes active request bytes, active prompt-message bytes and elapsed active time. The snapshot also records per-lane affinity-session count plus capability-compatible queued request count/bytes and oldest compatible queue age. These additions are observational only and do not change the production placement key. Until engine-truth cache overlap is exported, reusable-prefix/new-prefill work is explicitly labeled as `routing_history_exact_message_prefix_bytes_v1`: the supervisor hashes canonical messages and compares only identical leading history, retaining byte counts rather than prompt text. This is an experiment-only approximation, not a claim about tokenizer blocks or resident KV. Bench clients may additionally supply measured token counts and experiment labels through `X-Strata-Benchmark-Input-Tokens`, `X-Strata-Benchmark-Reusable-Prefix-Tokens`, `X-Strata-Benchmark-New-Prefill-Tokens`, `X-Strata-Benchmark-Output-Target-Tokens`, `X-Strata-Benchmark-Workload`, `X-Strata-Benchmark-Cache-State`, and `X-Strata-Benchmark-Interference-Arm`.

The proxy also exposes `X-Strata-Lane-Index`, `X-Strata-Admission-Rank`, `X-Strata-Queue-Wait-Ms` and an echoed benchmark request ID on generation responses. `/__multigpu/status` exposes current new-session queue depth/bytes, affinity/vision waiter counts and busy-lane count so an external sampler can retain queue-depth trajectories without reaching into scheduler internals. Persistent JSONL and human-readable console output are independent opt-ins: `--bench-trace-jsonl FILE` stores the structured trace, while `--bench-console-summary` prints one concise privacy-safe line per completed lease to stdout (and therefore to a tmux-attached runtime console) without creating the JSONL trace. The console line includes lane, selection reason, queue wait, streaming TTFT when available, E2E, HTTP status and completion result, and deliberately omits prompt content, request/session identifiers and affinity hashes.

`balanced-additive-new-prefill-retained-state-proxy-v1` is the promoted Phase-2B default, while `safe-affinity-live-state-v1` remains an explicit rollback control; either may run without persistent benchmark tracing. Other placement policies remain experimental and require `--bench-trace-jsonl` when selected through `--bench-scheduler-policy`. All policies preserve existing-session affinity, health/capability filtering, one active generation per lane, affinity reservations, vision reservation and compatible FIFO ordering; they can only choose differently among idle lanes for a new session. The trace retains the historical `placement_key` / `selected_placement_key` fields for compatibility and adds the active `placement_score` / `selected_placement_score`, policy name, scope and proxy source. Policies that use the last completed request byte footprint say `retained-state-proxy` in their names and report `retained_last_completed_request_bytes_v1`; this is routing-state evidence, not engine-truth active load or compute cost. The promoted balanced-additive policy orders new-session candidates first by remembered affinity-session count and then by additive estimated new-prefill plus retained-state cost. Persistent-supervisor validation on the reference host kept all six measured waves at 2/2/2 session placement across two independent campaigns, while the safe and unbalanced additive controls each developed 6-to-1 concentration on later waves; exact measurements are retained in the public recipe repository.

## Promoted Strata 0.1.39 baseline

Upstream Strata **0.1.39** (`6f32ec070f23ced9f50e704d854d775da52591ab`) was validated on `sync/upstream-0.1.39`, then promoted through PR #24 to main at merge commit `9ae0839b9f376dca9742804924db05902f872ad8`. The final pre-merge sync head was `1a454ca80d698af863693c1b2d3fafac0b3b1ceb`.

The promoted independent-lane baseline keeps one ordinary whole-GPU engine per lane. Because upstream 0.1.39 adds engine-internal request batching and layer-split batch groups, the supervisor removes inherited top-level `parallel` and CLI `--batch`, `--slots`, `--batch-groups`, and `--trim-stage-weights` from ordinary lane configs, in addition to the existing layer-split/peer stripping. This prevents a copied upstream config from silently changing the Lanes concurrency unit from one request/session per independent engine into nested batching. A future hybrid-lane experiment must opt into that coupling explicitly.

The supervisor also treats `/v1/responses` as generation work so the OpenAI Responses API follows the same admission, busy-lane exclusion, and lane selection path as Chat Completions and Anthropic Messages. Parking telemetry no longer extends the engine `DONE` record in this sync; upstream 0.1.39's native conversation-cache monitor owns that field space, avoiding the former semantic collision with upstream's appended offloaded-expert counter.

The bounded 0.1.39 compatibility gate passed **333 Python serving tests with 7 skipped**, a fresh **CUDA 13.4 sm_120 Release `strata` build** with `STRATA_BUILD_TESTS=OFF`, a second fresh Release build with `STRATA_BUILD_TESTS=ON` that linked the full `strata` executable, and focused `pinned_shared_test`, `expert_profile_save_test`, and `file_expert_source_test` CTests. A live IQ3_XXS single-lane smoke generated successfully on GPU 0. A live three-lane 32K compatibility smoke then routed three simultaneous requests one each to GPUs 0/1/2; lane 0 populated the ~39.97 GiB shared expert arena once, while lanes 1 and 2 reported `shared population ready; source load skipped`. These were the promotion/compatibility checks; they remain regression evidence distinct from the topology performance campaign.

A later bounded reference-host topology campaign exercised the 0.1.39 execution challenger explicitly. Ordinary lanes still strip nested batch/layer-split flags by default, but a separate upstream-native three-GPU layer-split server using explicit split `18,34`, `--batch 3 --batch-groups 3 --trim-stage-weights`, and the same shared expert backing measured **147.84 ± 2.26 tok/s** at M=2 versus **143.19 ± 3.06** for two lanes and **209.66 ± 4.02 tok/s** at M=3 versus **192.16 ± 4.11** for three lanes. Cold-prefill crosses by length: lanes lead at ~15K x3 (**5901.34 vs 3289.19 tok/s**), while the pipelined split narrowly leads at ~110K x3 (**6028.09 vs 5822.71 tok/s**). This is challenger evidence, not a change to the ordinary lane contract; details and caveats are in [`strata-0.1.39-performance-crossover-20261005.md`](strata-0.1.39-performance-crossover-20261005.md).

## Previous promoted engine baseline — Strata 0.1.38 (historical)

Before the 0.1.39 promotion, the shared-lane runtime used upstream Strata **0.1.38**, integrated from upstream commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`. The reference independent-lane contract—one engine per GPU, 262144 context and 32768 resident KV per lane, disjoint CPU partitions, asymmetric per-lane PCIe tuning, and one upstream-native shared expert arena—continues on 0.1.39 unless an explicit topology challenger is selected.

The 0.1.38 sync absorbs upstream 0.1.35-0.1.38 work including faster prompt/decode paths, silent-engine recovery, steadier PCIe probing, expert-profile persistence, unbuffered expert loading, service security/status changes, and the opt-in peer expert tier. The fork keeps its independent-lane supervisor, leader/follower shared-arena population, adaptive tracing, exact benchmark lease tracing, and session-aware scheduler. Multi-lane configs strip inherited `--layer-split` / `--split-device`, `--peer-*`, `--expert-cache-device1..3`, remote expert placement, and both CLI and top-level shared expert-profile writer settings so a copied parent config cannot silently expand one lane across multiple GPUs or create multiple writers for one profile file. A one-lane supervisor keeps upstream expert-profile persistence because it has only one writer.

The shared-arena implementation is no longer a fork-owned mmap wrapper: the supervisor passes upstream's native `--shared-expert-arena` option to each lane and adds the follower flag only to later sequential lanes. It explicitly forces `--conversation-cache-mib 0` until parked-conversation locality is modeled, preflights public/private ports before expensive model loading so stale listeners cannot satisfy readiness for a new lane, and owns a supervisor-level arena lock so concurrent population of one backing fails closed.

The bounded 0.1.38 compatibility gate passed **256 Python serving tests with 7 skipped**, a **CUDA 13.4 sm_120 Release `strata` build**, and focused `pinned_shared_test`, `expert_profile_save_test`, and `file_expert_source_test` CTests. A live model-backed three-lane smoke brought all three 262K lanes up on the shared arena and routed one simultaneous request to each GPU successfully. A matched single-lane A/B against the installed 0.1.31 binary measured ~15K prefill at **2498.4 → 2735.0 tok/s (+9.5%)** and ~30K at **2594.7 → 2784.0 tok/s (+7.3%)**; the short decode samples were acceptance-sensitive and are not promoted as a decode-performance claim. The latest full live scheduler/lifecycle campaign remains the retained 0.1.31 evidence, and the broader architecture matrix remains the retained 0.1.30 evidence. Those historical measurements are not relabeled as 0.1.38.

Adaptive hot-expert replacement remains engine-local to each lane. Optional `STRATA_ADAPT_TRACE` instrumentation records first routed misses, adaptive swap selection/publication, and the first later GPU-resident hit without changing the default serving path when tracing is disabled. Reference-host timing and A/B results live in the public recipe repository.

Controlled reference-host evidence for the architecture also includes **1→2→3 lane scaling, private-vs-shared arena PSS, and mixed RTX 5070 Ti + RTX 5060 Ti isolation**. The measurements, conditions, and caveats live in [`docs/systems-ablation-20260929.md`](https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe/blob/main/docs/systems-ablation-20260929.md); this implementation document intentionally does not duplicate hardware-specific result tables.

## Compatibility and limitations

The supervisor proxies Strata's OpenAI-compatible generation endpoints and preserves streaming. Functional serving behavior is covered by repository tests and reusable probes; concrete reference-host soak counts belong in the recipe repository.

Current boundaries:

1. Shared-arena mode is Linux-only.
2. Host-KV capacity is statically configured per lane.
3. Shared-arena startup uses one source expert population per supervisor generation; later lanes still perform their own dense/model-local initialization and GPU cache fill.
4. Hot-expert caches are lane-local; there is no required cross-GPU ownership scheme.
5. The supervisor is focused on generation serving and may route other endpoints through one lane.
6. Session affinity is only as strong as the available identity. Explicit session/conversation/thread IDs are authoritative; the first-user-message fallback is best effort and can change if a client rewrites or compacts away the first user turn.
7. Context lengths beyond the model/runtime's validated range remain experimental.
