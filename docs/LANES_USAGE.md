# Strata-Lanes experimental / reproduction usage

> **Research-mode notice:** this document explains how to reproduce the historical Strata-Lanes runtime. It is not a recommendation to use this fork instead of upstream Strata for normal serving. For ordinary use, prefer https://github.com/Niko1221/Strata.

This document covers the **Strata-Lanes multi-lane serving layer**. The serving entry point is `serve/multigpu_server.py`.

## 1. Start from a working single-GPU Strata config

Before using Lanes, make sure the same model/config works as an ordinary single-GPU Strata server on every GPU you intend to use.

The supervisor derives one private lane config per selected GPU. Ordinary lanes intentionally strip engine-internal multi-GPU options such as layer split, peer-device, and remote expert-helper flags so each lane owns exactly one physical GPU.

## 2. Launch the multi-lane supervisor

Example for three independent GPUs:

```bash
python3 serve/multigpu_server.py \
  --config /path/to/strata.json \
  --gpus 0,1,2 \
  --host 127.0.0.1 \
  --port 18086 \
  --base-port 19086 \
  --lane-contexts 262144,262144,262144 \
  --kv-budget 786432 \
  --lane-cpu-cores 5,6,5 \
  --lane-pcie-fracs 0.55,0.25,0.55 \
  --lane-kv-residents 32768,32768,32768 \
  --lane-vram-reserve-mibs 1200,1200,1200 \
  --vision-lanes 1 \
  --arena-file /dev/shm/strata-lanes.shared
```

The public OpenAI/Anthropic-compatible API is exposed on `--port`. Private lane servers use consecutive ports starting at `--base-port`.

The CPU, PCIe, context, KV and VRAM values above are **reference-host examples**, not portable defaults.

## 3. Preserve session affinity

A conversation should present the same stable identifier on every turn. The recommended header is:

```http
X-Strata-Session-Id: my-chat-123
```

Accepted affinity headers are:

```text
X-Strata-Session-Id
X-Conversation-Id
X-Session-Id
X-Thread-Id
```

The request body or `metadata` may alternatively contain one of:

```text
conversation_id
session_id
thread_id
```

If none is present, Lanes uses the first user message as a best-effort stable seed. Explicit IDs are preferred for agents and long-lived conversations because they make same-lane routing deterministic.

A known session waits for its remembered lane instead of spilling to another GPU. New sessions are placed only on eligible idle lanes.

## 4. Enable lane-local conversation parking

Conversation parking itself is an **upstream Strata engine feature**. Lanes adds bounded supervisor forwarding so each ordinary lane can use it safely with same-lane affinity.

Enable it on every lane:

```bash
  --experimental-conversation-cache-mib 4096 \
  --experimental-conversation-cache-slots 4 \
  --experimental-conversation-cache-min-free-mib 8192
```

The generated ordinary lane engines receive the upstream options:

```text
--conversation-cache-mib 4096
--conversation-cache-slots 4
--conversation-cache-min-free-mib 8192
```

Meaning:

- `mib`: maximum parked-conversation bytes per lane;
- `slots`: maximum parked conversation snapshots per lane;
- `min-free-mib`: host `MemAvailable` floor used by parking admission.

`slots` is **not** GPU count and is **not** request queue depth. Large agent prompts can hit the MiB budget before all slots are occupied.

Disable parking immediately with:

```bash
--experimental-conversation-cache-mib 0
```

An evicted snapshot is only a performance miss. The session remains owned by its lane and safely falls back to partial or full prompt recomputation.

## 5. Vision lanes

If the base Strata config contains a vision configuration, select which Lanes may accept image requests:

```bash
--vision-lanes 1
```

Text requests may still use that lane. Vision requests are restricted to vision-capable lanes without blocking unrelated text lanes.

## 6. Observe the supervisor

Supervisor status:

```bash
curl http://127.0.0.1:18086/__multigpu/status
```

Important per-lane fields include:

```text
alive
routable
busy
affinity_sessions
parked_conversations
parked_bytes
park_evictions
```

Useful response headers include:

```text
X-Strata-Lane-Index
X-Strata-Queue-Wait-Ms
X-Strata-Admission-Rank
```

For benchmark-grade server-side queue/admission records, launch with:

```bash
--bench-trace-jsonl /path/to/trace.jsonl
```

The trace is intended for experiments and observability; it is not required for ordinary serving.

## 7. Recovery behavior

The serving contract is:

- one active generation per lane;
- stable session affinity;
- new work avoids a lane whose child engine is unavailable;
- if a lane wrapper survives but its Strata child dies, an existing affinity may remain owned by that lane and its returning request can trigger the wrapper's child restart path;
- parking state is process-local optimization state, so a restarted child recomputes rather than assuming its old snapshot survived;
- client disconnect/cancellation releases the lane after cleanup.

## 8. Stop the server

Stop the `multigpu_server.py` process using your normal process/service manager. The supervisor owns its child lane servers and shared-arena lifecycle for that supervisor generation.

## Reference recipe

A measured 3× RTX 5070 Ti example, launch script and benchmark evidence are maintained separately:

https://github.com/rhgo1749/qwen3.8-flash-next-strata-gpu-per-lane-recipe