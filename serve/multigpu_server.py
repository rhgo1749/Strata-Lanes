"""Experimental 3-GPU Strata supervisor with a shared host expert arena.

This keeps Strata's existing single-GPU engine intact and composes several engine/server
processes into independent request lanes:

* each lane sees exactly one CUDA device (CUDA_VISIBLE_DEVICES)
* the resident expert arena is backed by upstream Strata's --shared-expert-arena MAP_SHARED file
* each lane keeps its own dense/QSA/MTP weights, hot-expert VRAM cache, and KV-resident window
* the full KV stays in host RAM through Strata's existing --kv-resident path
* PLE remains Strata's existing direct/O_DIRECT SSD reader

`--kv-budget` is a total capacity guard.  The first implementation partitions that
capacity between lanes; it is not yet a live cross-lane paged-KV allocator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
try:
    import fcntl
except ImportError:  # pragma: no cover - shared-arena multi-process mode is Linux-only
    fcntl = None
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
GENERATE_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/responses"}
VISION_METADATA_PATHS = {"/health", "/props", "/models", "/v1/models"}
AFFINITY_HEADERS = ("x-strata-session-id", "x-conversation-id", "x-session-id", "x-thread-id")
AFFINITY_FIELDS = ("conversation_id", "session_id", "thread_id")


def option_value(args: list[str], name: str) -> str | None:
    try:
        i = args.index(name)
    except ValueError:
        return None
    return args[i + 1] if i + 1 < len(args) else None


def replace_option(args: list[str], name: str, value: int | str) -> list[str]:
    out = list(args)
    try:
        i = out.index(name)
    except ValueError:
        out += [name, str(value)]
        return out
    if i + 1 >= len(out):
        raise ValueError(f"{name} has no value in the base config")
    out[i + 1] = str(value)
    return out


def remove_option(args: list[str], name: str) -> list[str]:
    """Remove every `name value` pair; malformed trailing occurrences are removed too."""
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == name:
            i += 2 if i + 1 < len(args) else 1
            continue
        out.append(args[i])
        i += 1
    return out


def remove_flag(args: list[str], name: str) -> list[str]:
    """Remove every standalone flag occurrence."""
    return [arg for arg in args if arg != name]


def sanitize_lane_config(cfg: dict, *, allow_profile_persistence: bool = False,
                         conversation_cache_mib: int = 0, conversation_cache_slots: int | None = None,
                         conversation_cache_min_free_mib: int | None = None) -> dict:
    """Return a lane-local config that cannot re-expand into an upstream multi-GPU engine."""
    lane_cfg = copy.deepcopy(cfg)
    lane_cfg.pop("gpu", None)
    lane_cfg.pop("layer_split", None)
    lane_cfg.pop("parallel", None)
    if not allow_profile_persistence:
        # server.py can synthesize the CLI writer flags from these top-level
        # keys, so stripping only cfg["args"] would still leave several lanes
        # writing the same learned-profile path.
        lane_cfg.pop("expert_profile_save", None)
        lane_cfg.pop("expert_profile_save_every", None)
    if isinstance(lane_cfg.get("args"), list):
        args = list(lane_cfg["args"])
        # Ordinary Lanes own exactly one physical GPU.  Upstream 0.1.38 adds
        # several opt-in ways for one engine to consume extra GPUs; strip all
        # inherited forms here so only a future explicit Super-Lane backend can
        # re-enable them.
        for name in (
            "--layer-split",
            "--split-device",
            "--peer-device",
            "--peer-reserve-mib",
            "--peer-slots",
            "--peer-adapt-swaps",
            "--peer-prefill-rows",
            "--expert-cache-device1",
            "--expert-cache-device2",
            "--expert-cache-device3",
            "--expert-cache-remote-placement",
            "--batch",
            "--slots",
            "--batch-groups",
        ):
            args = remove_option(args, name)
        args = remove_flag(args, "--split-skip-if-fits")
        args = remove_flag(args, "--trim-stage-weights")

        # Upstream 0.1.36 can persist an adaptive expert profile.  A copied
        # parent path would give several lane processes the same writer, so
        # keep persistence disabled for multi-lane operation until Lanes
        # assigns lane-local ownership.  A one-lane supervisor has one writer
        # and keeps the upstream behavior.
        if not allow_profile_persistence:
            args = remove_option(args, "--expert-profile-save")
            args = remove_option(args, "--expert-profile-save-every")

        # Production remains parking-off.  Experimental supervisor gates may
        # opt into bounded same-lane parking while strict affinity keeps each
        # conversation owned by the lane that first received it.  An engine
        # cache miss/eviction remains a normal prompt-recompute fallback.
        args = replace_option(args, "--conversation-cache-mib", max(0, conversation_cache_mib))
        if conversation_cache_slots is not None:
            args = replace_option(args, "--conversation-cache-slots", conversation_cache_slots)
        if conversation_cache_min_free_mib is not None:
            args = replace_option(args, "--conversation-cache-min-free-mib", conversation_cache_min_free_mib)
        lane_cfg["args"] = args
    return lane_cfg


def bind_lane_gpu(lane_cfg: dict, gpu: str) -> dict:
    """Bind a sanitized lane config to one physical GPU for engine launch and telemetry."""
    lane_cfg = copy.deepcopy(lane_cfg)
    lane_cfg["gpu"] = int(gpu)
    return lane_cfg


def apply_vision_capability(lane_cfg: dict, enabled: bool) -> dict:
    """Strip encoder config and its reserved VRAM from lanes that are not vision-capable."""
    if enabled or not lane_cfg.get("vision"):
        return lane_cfg
    lane_cfg.pop("vision", None)
    if isinstance(lane_cfg.get("args"), list):
        lane_cfg["args"] = remove_flag(lane_cfg["args"], "--vision")
        lane_cfg["args"] = remove_option(lane_cfg["args"], "--vram-reserve-mib")
    return lane_cfg


def apply_shared_arena(lane_cfg: dict, arena_file: Path | None, *, follower: bool = False) -> dict:
    """Use the explicit shared arena and mark only later sequential lanes as safe followers."""
    lane_cfg = copy.deepcopy(lane_cfg)
    args = remove_option(list(lane_cfg.get("args") or []), "--shared-expert-arena")
    args = remove_flag(args, "--shared-expert-arena-follower")
    if arena_file is not None:
        args += ["--shared-expert-arena", str(arena_file)]
        if follower:
            args += ["--shared-expert-arena-follower"]
    lane_cfg["args"] = args
    return lane_cfg


def resolve_config_path(value: str, cwd: str | None) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(cwd or ROOT) / p
    return p.resolve()


@dataclass(frozen=True)
class ArenaSpec:
    bytes: int
    expert_bytes: int
    max_blob: int
    n_expert: int


def native_arena_spec(pack: Path) -> ArenaSpec:
    """Return exactly the allocation ArenaExpertSource asks PinnedArena for."""
    meta = pack / "native_experts.txt"
    text = meta.read_text(encoding="utf-8")
    m = re.search(r"\bn_expert\s+(\d+)", text)
    n_expert = int(m.group(1)) if m else 512
    total = 0
    max_blob = 0
    rows = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 5:
            raise ValueError(f"{meta}: malformed native expert row: {raw!r}")
        off, blob = int(f[3]), int(f[4])
        if off < 0 or blob <= 0:
            raise ValueError(f"{meta}: invalid offset/blob in row: {raw!r}")
        total = max(total, off + blob * n_expert)
        max_blob = max(max_blob, blob)
        rows += 1
    if rows == 0 or total == 0 or max_blob == 0:
        raise ValueError(f"{meta}: no native expert rows")
    return ArenaSpec(total + max_blob, total, max_blob, n_expert)


def parse_int_list(text: str, *, what: str) -> list[int]:
    try:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{what} must be a comma-separated integer list") from e
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{what} must contain positive integers")
    return values


def parse_float_list(text: str, *, what: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{what} must be a comma-separated numeric list") from e
    if not values or any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError(f"{what} values must be between 0 and 1")
    return values


def parse_lane_indices(text: str, lanes: int, *, what: str) -> set[int]:
    if text.strip().lower() == "none":
        return set()
    try:
        values = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{what} must be a comma-separated lane-index list or 'none'") from e
    if not values:
        raise ValueError(f"{what} must name at least one lane or 'none'")
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must not contain duplicate lane indices")
    if any(v < 0 or v >= lanes for v in values):
        raise ValueError(f"{what} lane indices must be between 0 and {lanes - 1}")
    return set(values)


def request_has_images(body: bytes) -> bool:
    """Classify supported OpenAI/Anthropic message bodies without interpreting image contents."""
    try:
        req = json.loads(body) if body else {}
    except (TypeError, ValueError):
        return False
    messages = req.get("messages") if isinstance(req, dict) else None
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                return True
    return False


def _affinity_digest(source: str, value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((source + "\0" + canonical).encode("utf-8")).hexdigest()


def request_affinity_key(body: bytes, headers=None) -> str | None:
    """Return a privacy-safe stable key for a conversation when the request exposes one.

    Explicit conversation/session/thread identifiers win.  For ordinary OpenAI/Anthropic
    chat bodies that do not carry one, the first user message is a best-effort stable seed:
    appended history then keeps the same lane and its lane-local conversation/KV state.
    """
    if headers is not None:
        for name in AFFINITY_HEADERS:
            value = headers.get(name)
            if value not in (None, ""):
                return _affinity_digest("header:" + name, value)
    try:
        req = json.loads(body) if body else {}
    except (TypeError, ValueError):
        return None
    if not isinstance(req, dict):
        return None
    for name in AFFINITY_FIELDS:
        value = req.get(name)
        if isinstance(value, (str, int)) and value != "":
            return _affinity_digest("body:" + name, value)
    metadata = req.get("metadata")
    if isinstance(metadata, dict):
        for name in AFFINITY_FIELDS:
            value = metadata.get(name)
            if isinstance(value, (str, int)) and value != "":
                return _affinity_digest("metadata:" + name, value)
    messages = req.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            seed = {"content": message.get("content")}
            if message.get("name") is not None:
                seed["name"] = message.get("name")
            return _affinity_digest("first-user", seed)
    return None


def request_prompt_signature(body: bytes) -> tuple[tuple[str, int], ...]:
    """Return privacy-safe exact-message prefix units for routing-history reuse estimates.

    This is deliberately not called a token/KV-cache measurement.  Each message is
    canonicalized, hashed, and paired with its UTF-8 byte length.  Matching leading
    units therefore estimate how much conversation history is identical to the last
    request served by a lane without retaining prompt text in the supervisor.
    """
    try:
        req = json.loads(body) if body else {}
    except (TypeError, ValueError):
        return ()
    if not isinstance(req, dict):
        return ()
    messages = req.get("messages")
    if not isinstance(messages, list):
        return ()
    out: list[tuple[str, int]] = []
    for message in messages:
        canonical = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        encoded = canonical.encode("utf-8")
        out.append((hashlib.sha256(encoded).hexdigest(), len(encoded)))
    return tuple(out)


def reusable_prefix_bytes(current: tuple[tuple[str, int], ...],
                          previous: tuple[tuple[str, int], ...]) -> int:
    """Count canonical message bytes in the exact common leading history."""
    total = 0
    for (current_hash, current_bytes), (previous_hash, _previous_bytes) in zip(current, previous):
        if current_hash != previous_hash:
            break
        total += current_bytes
    return total


def request_is_streaming(body: bytes) -> bool:
    try:
        req = json.loads(body) if body else {}
    except (TypeError, ValueError):
        return False
    return isinstance(req, dict) and req.get("stream") is True


def lane_contexts(base_context: int, lanes: int, requested: str | None, kv_budget: int | None) -> list[int]:
    if requested:
        values = parse_int_list(requested, what="--lane-contexts")
        if len(values) != lanes:
            raise ValueError(f"--lane-contexts has {len(values)} values for {lanes} GPU lanes")
    elif kv_budget is not None:
        q, r = divmod(kv_budget, lanes)
        values = [q + (1 if i < r else 0) for i in range(lanes)]
    else:
        values = [base_context] * lanes
    if kv_budget is not None and sum(values) > kv_budget:
        raise ValueError(
            f"lane contexts total {sum(values)} tokens, above --kv-budget {kv_budget}"
        )
    return values


def physical_core_cpu_sets() -> list[tuple[int, ...]]:
    """Allowed logical CPUs grouped by physical core, in the process's current affinity order."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    groups: dict[tuple[int, int], list[int]] = {}
    for cpu in allowed:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topo / "physical_package_id").read_text().strip())
            core = int((topo / "core_id").read_text().strip())
            key = (package, core)
        except (OSError, ValueError):
            key = (0, cpu)
        groups.setdefault(key, []).append(cpu)
    return [tuple(v) for v in groups.values()]


def partition_cpu_sets(core_groups: list[tuple[int, ...]], lanes: int) -> list[tuple[int, ...]]:
    """Round-robin physical cores across lanes so SMT siblings always stay together."""
    if lanes < 1:
        raise ValueError("lane count must be positive")
    if len(core_groups) < lanes * 2:
        raise ValueError(
            f"--cpu-partition auto needs at least two physical cores per lane; "
            f"found {len(core_groups)} cores for {lanes} lanes"
        )
    out: list[list[int]] = [[] for _ in range(lanes)]
    for i, group in enumerate(core_groups):
        out[i % lanes].extend(group)
    return [tuple(sorted(x)) for x in out]


def partition_cpu_sets_exact(core_groups: list[tuple[int, ...]], counts: list[int]) -> list[tuple[int, ...]]:
    """Spread an exact physical-core budget across lanes while keeping SMT siblings together."""
    if not counts or any(x < 2 for x in counts):
        raise ValueError("--lane-cpu-cores needs at least two physical cores per lane")
    if sum(counts) != len(core_groups):
        raise ValueError(
            f"--lane-cpu-cores totals {sum(counts)} physical cores but {len(core_groups)} are available"
        )
    out: list[list[int]] = [[] for _ in counts]
    assigned = [0] * len(counts)
    credit = [0] * len(counts)
    total = sum(counts)
    for group in core_groups:
        for i, weight in enumerate(counts):
            if assigned[i] < weight:
                credit[i] += weight
        candidates = [i for i in range(len(counts)) if assigned[i] < counts[i]]
        lane = max(candidates, key=lambda i: (credit[i], -i))
        credit[lane] -= total
        assigned[lane] += 1
        out[lane].extend(group)
    return [tuple(sorted(x)) for x in out]


def _set_process_affinity(cpus: tuple[int, ...]) -> None:
    os.sched_setaffinity(0, set(cpus))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_head(root: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def default_state_dir(config: Path, gpus: list[str]) -> Path:
    key = hashlib.sha256(
        (str(config.resolve()) + "\0" + ",".join(gpus)).encode("utf-8")
    ).hexdigest()[:16]
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "strata" / "multigpu" / key


class SharedArenaLease:
    """Hold one supervisor-level ownership lock for a shared-arena pathname."""

    def __init__(self, arena_file: Path):
        if fcntl is None:
            raise RuntimeError("shared-arena supervisor ownership requires Linux fcntl/flock")
        self.path = Path(str(arena_file) + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.file.close()
            raise RuntimeError(f"shared arena is already owned by another supervisor: {arena_file}") from e

    def close(self) -> None:
        if self.file is None:
            return
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None


def default_arena_file(pack: Path, spec: ArenaSpec) -> Path:
    key = hashlib.sha256(
        (str(pack.resolve()) + f"\0{spec.expert_bytes}\0{spec.max_blob}").encode("utf-8")
    ).hexdigest()[:16]
    # Upstream 0.1.30's file-backed arena is intended for tmpfs.  Keep the
    # default off persistent storage; --arena-file remains available for an
    # operator-selected tmpfs path.
    return Path("/dev/shm") / "strata" / "shared-arena" / f"{pack.name}-{key}.bin"


@dataclass
class Lane:
    index: int
    gpu: str
    port: int
    context: int
    config: Path
    cpus: tuple[int, ...] | None = None
    pcie_frac: float | None = None
    kv_resident: int | None = None
    vision: bool = False
    vram_reserve_mib: int | None = None
    process: subprocess.Popen | None = None
    busy: bool = False
    live_affinity_key: str | None = None
    live_request_bytes: int = 0
    live_sequence: int = 0
    live_prompt_signature: tuple[tuple[str, int], ...] = ()
    active_affinity_key: str | None = None
    active_request_bytes: int = 0
    active_prompt_signature: tuple[tuple[str, int], ...] = ()
    active_started_mono_ns: int | None = None


@dataclass(frozen=True)
class Waiter:
    requires_vision: bool
    request_bytes: int
    entered_mono_ns: int


@dataclass
class ProxyObservation:
    status: int | None = None
    first_byte_mono_ns: int | None = None
    first_byte_unix_ns: int | None = None
    response_bytes: int = 0
    completion_reason: str = "proxy_error"
    error: str | None = None


class BenchmarkTrace:
    """Append exact supervisor lease timings without changing normal serving behavior."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)


def benchmark_console_summary(record: dict) -> str:
    """Human-readable one-line benchmark summary without prompt/session identifiers."""

    def ms(value) -> str:
        return "-" if value is None else f"{float(value):.1f}ms"

    scheduler = record.get("scheduler") if isinstance(record.get("scheduler"), dict) else {}
    return (
        "[strata-multigpu][bench] "
        f"lane={record.get('lane_index', '-')} "
        f"reason={scheduler.get('selected_reason', '-')} "
        f"queue={ms(record.get('queue_wait_ms'))} "
        f"ttft={ms(record.get('ttft_ms'))} "
        f"e2e={ms(record.get('e2e_ms'))} "
        f"status={record.get('http_status', '-')} "
        f"result={record.get('completion_reason', '-')}"
    )


def lane_service_ready(lane: Lane, timeout: float = 0.2) -> bool:
    """Return whether the lane wrapper is healthy enough to accept a request."""
    if lane.process is None or lane.process.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{lane.port}/health", timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def lane_engine_alive(lane: Lane, timeout: float = 0.2) -> bool:
    """Return whether both the lane wrapper and its child engine currently expose the model."""
    if lane.process is None or lane.process.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{lane.port}/v1/models", timeout=timeout) as r:
            if not (200 <= r.status < 300):
                return False
            payload = json.loads(r.read() or b"{}")
            return bool(payload.get("data"))
    except Exception:
        return False


SCHEDULER_POLICY_SAFE = "safe-affinity-live-state-v1"
SCHEDULER_POLICY_DEFAULT = "balanced-additive-new-prefill-retained-state-proxy-v1"
SCHEDULER_POLICIES = (
    SCHEDULER_POLICY_SAFE,
    "round-robin-idle-v1",
    "retained-state-smallest-v1",
    "cache-aware-idle-v1",
    "additive-new-prefill-retained-state-proxy-v1",
    "multiplicative-new-prefill-retained-state-proxy-v1",
    "session-start-balance-cache-aware-v1",
    SCHEDULER_POLICY_DEFAULT,
)
SCHEDULER_POLICIES_WITHOUT_TRACE = frozenset({SCHEDULER_POLICY_SAFE, SCHEDULER_POLICY_DEFAULT})


class LanePool:
    def __init__(self, lanes: list[Lane], *, max_affinity_entries: int | None = None,
                 scheduler_policy: str = SCHEDULER_POLICY_DEFAULT):
        if scheduler_policy not in SCHEDULER_POLICIES:
            raise ValueError(f"unknown scheduler policy: {scheduler_policy}")
        self.lanes = lanes
        self.scheduler_policy = scheduler_policy
        self.cv = threading.Condition()
        self.cursor = 0
        self.vision_waiters = 0
        self.max_affinity_entries = max_affinity_entries or max(16, len(lanes) * 8)
        self.affinity: OrderedDict[str, int] = OrderedDict()
        self.session_turns: OrderedDict[str, int] = OrderedDict()
        self.affinity_waiters: dict[int, int] = {lane.index: 0 for lane in lanes}
        self.waiters: OrderedDict[int, Waiter] = OrderedDict()
        self.next_wait_ticket = 0
        self.sequence = 0

    def _remember_affinity(self, key: str, lane_index: int) -> None:
        self.affinity[key] = lane_index
        self.affinity.move_to_end(key)
        while len(self.affinity) > self.max_affinity_entries:
            old_key, _ = self.affinity.popitem(last=False)
            self.session_turns.pop(old_key, None)

    def _drop_dead_affinity(self, alive_indices: set[int]) -> None:
        for key, lane_index in list(self.affinity.items()):
            if lane_index not in alive_indices:
                del self.affinity[key]
                self.session_turns.pop(key, None)

    def _advance_session_turn(self, key: str | None) -> int | None:
        if key is None:
            return None
        turn = self.session_turns.get(key, 0) + 1
        self.session_turns[key] = turn
        self.session_turns.move_to_end(key)
        return turn

    def _register_waiter(self, requires_vision: bool, request_bytes: int) -> int:
        ticket = self.next_wait_ticket
        self.next_wait_ticket += 1
        self.waiters[ticket] = Waiter(requires_vision, max(0, request_bytes), time.monotonic_ns())
        return ticket

    def _drop_waiter(self, ticket: int | None) -> None:
        if ticket is not None:
            self.waiters.pop(ticket, None)

    def _start_lane_work(self, lane: Lane, *, affinity_key: str | None,
                         request_bytes: int,
                         prompt_signature: tuple[tuple[str, int], ...]) -> None:
        lane.active_affinity_key = affinity_key
        lane.active_request_bytes = max(0, request_bytes)
        lane.active_prompt_signature = prompt_signature
        lane.active_started_mono_ns = time.monotonic_ns()
        lane.busy = True

    def _compatible_waiter_stats(self, lane: Lane, now_ns: int) -> tuple[int, int, float | None]:
        compatible = [
            waiter for waiter in self.waiters.values()
            if not waiter.requires_vision or lane.vision
        ]
        if not compatible:
            return 0, 0, None
        oldest_ns = min(waiter.entered_mono_ns for waiter in compatible)
        return (
            len(compatible),
            sum(waiter.request_bytes for waiter in compatible),
            max(0.0, (now_ns - oldest_ns) / 1e6),
        )

    def _earlier_waiter_can_use(self, ticket: int, lane: Lane) -> bool:
        for other_ticket, waiter in self.waiters.items():
            if other_ticket == ticket:
                break
            if waiter.requires_vision and not lane.vision:
                continue
            if not waiter.requires_vision and lane.vision and self.vision_waiters:
                continue
            return True
        return False

    def _policy_components(self, lane: Lane, *,
                           prompt_signature: tuple[tuple[str, int], ...],
                           position: int) -> dict:
        prompt_bytes = sum(size for _digest, size in prompt_signature)
        reuse_bytes = reusable_prefix_bytes(prompt_signature, lane.live_prompt_signature)
        return {
            "estimated_reusable_prefix_bytes": reuse_bytes,
            "estimated_new_prefill_bytes": max(0, prompt_bytes - reuse_bytes),
            "retained_state_proxy_bytes": max(0, lane.live_request_bytes),
            "rotation_offset": (position - self.cursor) % len(self.lanes),
            "affinity_session_count": sum(
                1 for lane_index in self.affinity.values() if lane_index == lane.index
            ),
        }

    def _placement_score(self, lane: Lane, *,
                         prompt_signature: tuple[tuple[str, int], ...],
                         request_bytes: int,
                         position: int) -> tuple:
        c = self._policy_components(
            lane,
            prompt_signature=prompt_signature,
            position=position,
        )
        reuse = c["estimated_reusable_prefix_bytes"]
        new_prefill = c["estimated_new_prefill_bytes"]
        retained = c["retained_state_proxy_bytes"]
        rotation = c["rotation_offset"]

        if self.scheduler_policy == SCHEDULER_POLICY_SAFE:
            return (0 if retained == 0 else 1, retained, lane.live_sequence, rotation)
        if self.scheduler_policy == "round-robin-idle-v1":
            return (rotation,)
        if self.scheduler_policy == "retained-state-smallest-v1":
            return (retained, lane.live_sequence, rotation)
        if self.scheduler_policy == "cache-aware-idle-v1":
            if reuse:
                return (0, -reuse, retained, lane.live_sequence, rotation)
            return (1, 0 if retained == 0 else 1, retained, lane.live_sequence, rotation)
        if self.scheduler_policy == "additive-new-prefill-retained-state-proxy-v1":
            return (new_prefill + retained, lane.live_sequence, rotation)
        if self.scheduler_policy == "multiplicative-new-prefill-retained-state-proxy-v1":
            load_ratio = retained / max(1, request_bytes)
            return (new_prefill * (1.0 + load_ratio), lane.live_sequence, rotation)
        if self.scheduler_policy == "session-start-balance-cache-aware-v1":
            return (
                c["affinity_session_count"],
                -reuse,
                retained,
                lane.live_sequence,
                rotation,
            )
        if self.scheduler_policy == "balanced-additive-new-prefill-retained-state-proxy-v1":
            return (
                c["affinity_session_count"],
                new_prefill + retained,
                lane.live_sequence,
                rotation,
            )
        raise AssertionError(f"unhandled scheduler policy: {self.scheduler_policy}")

    def _decision_record(self, *, alive: list[Lane], eligible: list[Lane], selected: Lane,
                         selected_reason: str, request_bytes: int,
                         prompt_signature: tuple[tuple[str, int], ...],
                         session_turn: int | None) -> dict:
        alive_indices = {lane.index for lane in alive}
        eligible_indices = {lane.index for lane in eligible}
        prompt_bytes = sum(size for _digest, size in prompt_signature)
        now_ns = time.monotonic_ns()
        lane_components = []
        for position, lane in enumerate(self.lanes):
            policy_components = self._policy_components(
                lane,
                prompt_signature=prompt_signature,
                position=position,
            )
            reuse_bytes = policy_components["estimated_reusable_prefix_bytes"]
            rotation_offset = policy_components["rotation_offset"]
            queued_count, queued_bytes, oldest_queue_age_ms = self._compatible_waiter_stats(lane, now_ns)
            placement_key = [
                0 if lane.live_request_bytes == 0 else 1,
                lane.live_request_bytes,
                lane.live_sequence,
                rotation_offset,
            ]
            placement_score = list(self._placement_score(
                lane,
                prompt_signature=prompt_signature,
                request_bytes=request_bytes,
                position=position,
            ))
            lane_components.append({
                "lane_index": lane.index,
                "alive": lane.index in alive_indices,
                "eligible": lane.index in eligible_indices,
                "busy": lane.busy,
                "vision": lane.vision,
                "affinity_waiters": self.affinity_waiters.get(lane.index, 0),
                "live_request_bytes": lane.live_request_bytes,
                "retained_prompt_message_bytes": sum(
                    size for _digest, size in lane.live_prompt_signature
                ),
                "live_sequence": lane.live_sequence,
                "active_request_bytes": lane.active_request_bytes,
                "active_prompt_message_bytes": sum(
                    size for _digest, size in lane.active_prompt_signature
                ),
                "active_elapsed_ms": (
                    max(0.0, (now_ns - lane.active_started_mono_ns) / 1e6)
                    if lane.busy and lane.active_started_mono_ns is not None else None
                ),
                "compatible_queued_request_count": queued_count,
                "compatible_queued_request_bytes": queued_bytes,
                "oldest_compatible_queue_age_ms": oldest_queue_age_ms,
                **policy_components,
                "placement_key": placement_key,
                "placement_score": placement_score,
            })
        selected_component = next(x for x in lane_components if x["lane_index"] == selected.index)
        return {
            "policy": self.scheduler_policy,
            "selected_reason": selected_reason,
            "session_turn": session_turn,
            "request_bytes": max(0, request_bytes),
            "prompt_message_bytes": prompt_bytes,
            "reuse_estimate_source": "routing_history_exact_message_prefix_bytes_v1",
            "load_proxy_source": "retained_last_completed_request_bytes_v1",
            "policy_scope": "new_session_idle_lane_only",
            "active_lane_count": sum(1 for lane in alive if lane.busy),
            "queued_request_count": len(self.waiters),
            "queued_request_bytes": sum(waiter.request_bytes for waiter in self.waiters.values()),
            "affinity_waiter_count": sum(self.affinity_waiters.values()),
            "selected_placement_key": selected_component["placement_key"],
            "selected_placement_score": selected_component["placement_score"],
            "lane_components": lane_components,
        }

    def acquire(self, *, requires_vision: bool = False, affinity_key: str | None = None,
                request_bytes: int = 0,
                prompt_signature: tuple[tuple[str, int], ...] = (),
                decision_out: dict | None = None) -> Lane:
        with self.cv:
            wait_ticket: int | None = None
            reserved_lane_index: int | None = None
            if requires_vision:
                self.vision_waiters += 1
            try:
                while True:
                    alive = [x for x in self.lanes if lane_engine_alive(x)]
                    # New sessions use only loaded engines. A returning affinity may preserve its
                    # lane while the private wrapper is alive so that forwarding the request can
                    # trigger server.py's synchronous child-engine restart path.
                    restarting_affinity_lane = None
                    if affinity_key is not None and affinity_key in self.affinity:
                        candidate = self.lanes[self.affinity[affinity_key]]
                        if candidate not in alive and lane_service_ready(candidate):
                            restarting_affinity_lane = candidate
                    if not alive and restarting_affinity_lane is None:
                        raise RuntimeError("all GPU lanes have stopped")
                    # Affinity belongs to the lane wrapper/control plane, not to one child-engine
                    # process. Preserve it across a child crash so the returning session can trigger
                    # same-lane restart; purge only when the lane wrapper itself is gone.
                    wrapper_indices = {
                        x.index for x in self.lanes
                        if x.process is not None and x.process.poll() is None
                    }
                    self._drop_dead_affinity(wrapper_indices)
                    eligible = [x for x in alive if not requires_vision or x.vision]
                    restarting_eligible = (
                        restarting_affinity_lane is not None
                        and (not requires_vision or restarting_affinity_lane.vision)
                    )
                    if not eligible and not restarting_eligible:
                        raise RuntimeError("no vision-capable GPU lanes are running")

                    if affinity_key is not None and affinity_key in self.affinity:
                        if wait_ticket is not None:
                            self._drop_waiter(wait_ticket)
                            wait_ticket = None
                            self.cv.notify_all()
                        preferred = self.lanes[self.affinity[affinity_key]]
                        if requires_vision and not preferred.vision:
                            del self.affinity[affinity_key]
                            self.session_turns.pop(affinity_key, None)
                            if reserved_lane_index is not None:
                                self.affinity_waiters[reserved_lane_index] -= 1
                                reserved_lane_index = None
                                self.cv.notify_all()
                        else:
                            if preferred.busy or (not requires_vision and preferred.vision and self.vision_waiters):
                                if reserved_lane_index != preferred.index:
                                    if reserved_lane_index is not None:
                                        self.affinity_waiters[reserved_lane_index] -= 1
                                    self.affinity_waiters[preferred.index] += 1
                                    reserved_lane_index = preferred.index
                                self.cv.wait(timeout=1.0)
                                continue
                            if reserved_lane_index is not None:
                                self.affinity_waiters[reserved_lane_index] -= 1
                                reserved_lane_index = None
                            self.affinity.move_to_end(affinity_key)
                            session_turn = self._advance_session_turn(affinity_key)
                            if decision_out is not None:
                                decision_alive = alive if preferred in alive else [*alive, preferred]
                                decision_eligible = eligible if preferred in eligible else [*eligible, preferred]
                                decision_out.update(self._decision_record(
                                    alive=decision_alive,
                                    eligible=decision_eligible,
                                    selected=preferred,
                                    selected_reason="session_affinity_restart" if preferred not in alive else "session_affinity",
                                    request_bytes=request_bytes,
                                    prompt_signature=prompt_signature,
                                    session_turn=session_turn,
                                ))
                            self._start_lane_work(
                                preferred,
                                affinity_key=affinity_key,
                                request_bytes=request_bytes,
                                prompt_signature=prompt_signature,
                            )
                            self.cv.notify_all()
                            return preferred

                    if wait_ticket is None:
                        wait_ticket = self._register_waiter(requires_vision, request_bytes)
                    candidates: list[tuple[int, Lane]] = []
                    for offset in range(len(self.lanes)):
                        idx = (self.cursor + offset) % len(self.lanes)
                        lane = self.lanes[idx]
                        if lane not in eligible or lane.busy:
                            continue
                        if self.affinity_waiters.get(lane.index, 0) > 0:
                            continue
                        if not requires_vision and lane.vision and self.vision_waiters:
                            continue
                        if self._earlier_waiter_can_use(wait_ticket, lane):
                            continue
                        candidates.append((idx, lane))
                    if candidates:
                        idx, lane = min(
                            candidates,
                            key=lambda pair: self._placement_score(
                                pair[1],
                                prompt_signature=prompt_signature,
                                request_bytes=request_bytes,
                                position=pair[0],
                            ),
                        )
                        self._drop_waiter(wait_ticket)
                        wait_ticket = None
                        session_turn = self._advance_session_turn(affinity_key)
                        if decision_out is not None:
                            decision_out.update(self._decision_record(
                                alive=alive,
                                eligible=eligible,
                                selected=lane,
                                selected_reason=(
                                    "live_state_lexicographic"
                                    if self.scheduler_policy == SCHEDULER_POLICY_SAFE
                                    else (
                                        "session_start_balance_additive"
                                        if self.scheduler_policy == SCHEDULER_POLICY_DEFAULT
                                        else "benchmark_policy_score"
                                    )
                                ),
                                request_bytes=request_bytes,
                                prompt_signature=prompt_signature,
                                session_turn=session_turn,
                            ))
                        if affinity_key is not None:
                            self._remember_affinity(affinity_key, lane.index)
                        self._start_lane_work(
                            lane,
                            affinity_key=affinity_key,
                            request_bytes=request_bytes,
                            prompt_signature=prompt_signature,
                        )
                        self.cursor = (idx + 1) % len(self.lanes)
                        self.cv.notify_all()
                        return lane
                    self.cv.wait(timeout=1.0)
            finally:
                self._drop_waiter(wait_ticket)
                if reserved_lane_index is not None:
                    self.affinity_waiters[reserved_lane_index] -= 1
                if requires_vision:
                    self.vision_waiters -= 1
                self.cv.notify_all()

    def release(self, lane: Lane, *, affinity_key: str | None = None, request_bytes: int = 0,
                prompt_signature: tuple[tuple[str, int], ...] = ()) -> None:
        with self.cv:
            self.sequence += 1
            lane.live_affinity_key = affinity_key
            lane.live_request_bytes = max(0, request_bytes)
            lane.live_sequence = self.sequence
            lane.live_prompt_signature = prompt_signature
            lane.active_affinity_key = None
            lane.active_request_bytes = 0
            lane.active_prompt_signature = ()
            lane.active_started_mono_ns = None
            lane.busy = False
            self.cv.notify_all()

    def metadata_lane(self, fallback: Lane) -> Lane:
        for lane in self.lanes:
            if lane.vision and lane_engine_alive(lane):
                return lane
        return fallback

    def status(self) -> list[dict]:
        with self.cv:
            return [
                {
                    "index": x.index,
                    "gpu": x.gpu,
                    "port": x.port,
                    "context": x.context,
                    "cpus": list(x.cpus) if x.cpus else None,
                    "pcie_frac": x.pcie_frac,
                    "kv_resident": x.kv_resident,
                    "vision": x.vision,
                    "vram_reserve_mib": x.vram_reserve_mib,
                    "pid": x.process.pid if x.process else None,
                    "wrapper_alive": bool(x.process and x.process.poll() is None),
                    "alive": lane_engine_alive(x),
                    "routable": lane_service_ready(x),
                    "busy": x.busy,
                    "affinity_sessions": sum(1 for lane_index in self.affinity.values() if lane_index == x.index),
                    "affinity_waiters": self.affinity_waiters.get(x.index, 0),
                    "live_session": x.live_affinity_key[:12] if x.live_affinity_key else None,
                    "live_request_bytes": x.live_request_bytes,
                    "retained_prompt_message_bytes": sum(
                        size for _digest, size in x.live_prompt_signature
                    ),
                    "live_sequence": x.live_sequence,
                    "active_session": x.active_affinity_key[:12] if x.active_affinity_key else None,
                    "active_request_bytes": x.active_request_bytes,
                    "active_prompt_message_bytes": sum(
                        size for _digest, size in x.active_prompt_signature
                    ),
                    "active_elapsed_ms": (
                        max(0.0, (time.monotonic_ns() - x.active_started_mono_ns) / 1e6)
                        if x.busy and x.active_started_mono_ns is not None else None
                    ),
                }
                for x in self.lanes
            ]

    def queue_depth(self) -> int:
        with self.cv:
            return len(self.waiters)

    def queue_status(self) -> dict:
        with self.cv:
            return {
                "new_session_waiters": len(self.waiters),
                "queued_request_bytes": sum(waiter.request_bytes for waiter in self.waiters.values()),
                "affinity_waiters": sum(self.affinity_waiters.values()),
                "vision_waiters": self.vision_waiters,
                "busy_lanes": sum(1 for lane in self.lanes if lane.busy),
            }


def require_ports_free(host: str, ports: list[int]) -> None:
    """Fail before model loading only when a fixed port has a live listener.

    A bind-based probe also rejects a recently closed server while its socket is
    in TIME_WAIT, which makes back-to-back benchmark launches fail even though
    no stale service can answer readiness.  Probe connectability instead: a
    successful connect means there is an actual listener that could satisfy
    /health; connection-refused means the port is safe to reuse.
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    for port in ports:
        if port == 0:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        try:
            if s.connect_ex((probe_host, port)) == 0:
                raise RuntimeError(f"port {host}:{port} is already in use")
        finally:
            s.close()


def lane_parking_status(lane: Lane, timeout: float = 0.25) -> dict:
    """Best-effort engine-truth parking counters from the lane's latest completed request."""
    if not lane_engine_alive(lane):
        return {"parked_conversations": None, "parked_bytes": None, "park_evictions": None}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{lane.port}/metrics", timeout=timeout) as r:
            payload = json.loads(r.read())
        rows = payload.get("requests") or []
        latest = rows[0] if rows else {}
        return {
            "parked_conversations": latest.get("parked_conversations"),
            "parked_bytes": latest.get("parked_bytes"),
            "park_evictions": latest.get("park_evictions"),
        }
    except Exception:
        return {"parked_conversations": None, "parked_bytes": None, "park_evictions": None}


def wait_ready(lane: Lane, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{lane.port}/health"
    while time.monotonic() < deadline:
        if lane.process is not None and lane.process.poll() is not None:
            raise RuntimeError(
                f"lane {lane.index} (GPU {lane.gpu}) exited with code {lane.process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if 200 <= r.status < 300:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"lane {lane.index} (GPU {lane.gpu}) did not become ready")


def stop_lane(lane: Lane) -> None:
    p = lane.process
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)


def make_handler(pool: LanePool, lane0: Lane, arena_file: Path, arena_bytes: int, kv_budget: int | None,
                 reject_generate_proxy: bool = False, bench_trace: BenchmarkTrace | None = None,
                 bench_console_summary: bool = False):
    counter_lock = threading.Lock()
    request_counter = 0
    admission_counters: dict[str, int] = {}

    def next_request_id(explicit: str | None) -> str:
        nonlocal request_counter
        if explicit:
            return explicit
        with counter_lock:
            request_counter += 1
            return f"auto-{request_counter}"

    def next_admission_rank(run_id: str | None) -> int:
        key = run_id or "__unscoped__"
        with counter_lock:
            rank = admission_counters.get(key, 0)
            admission_counters[key] = rank + 1
            return rank

    def benchmark_header_int(headers, name: str) -> int | None:
        value = headers.get(name)
        if value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print("[strata-multigpu] " + (fmt % args), flush=True)

        def _json(self, status: int, value) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _status(self):
            lanes = pool.status()
            parking = {lane.index: lane_parking_status(lane) for lane in pool.lanes}
            for row in lanes:
                row.update(parking.get(row["index"], {}))
            self._json(200, {
                "status": "ok",
                "mode": "partitioned-multigpu-v1",
                "arena_file": str(arena_file),
                "arena_bytes": arena_bytes,
                "kv_budget": kv_budget,
                "lane_context_total": sum(x.context for x in pool.lanes),
                "queue_depth": pool.queue_depth(),
                "queue": pool.queue_status(),
                "bench_trace_jsonl": str(bench_trace.path) if bench_trace else None,
                "bench_console_summary": bench_console_summary,
                "scheduler_policy": pool.scheduler_policy,
                "lanes": lanes,
            })

        def _slots(self):
            self._json(200, [
                {"id": lane.index, "n_ctx": lane.context,
                 "is_processing": lane.busy, "gpu": lane.gpu, "vision": lane.vision}
                for lane in pool.lanes if lane_engine_alive(lane)
            ])

        def _body(self) -> bytes:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                raise ValueError("chunked request bodies are not supported by the multigpu proxy")
            n = int(self.headers.get("Content-Length", "0") or 0)
            return self.rfile.read(n) if n else b""

        def _proxy(self, lane: Lane, body: bytes | None = None,
                   extra_headers: dict[str, str] | None = None) -> ProxyObservation:
            observation = ProxyObservation()
            if body is None:
                try:
                    body = self._body()
                except ValueError as e:
                    observation.status = 411
                    observation.completion_reason = "request_rejected"
                    observation.error = str(e)
                    self._json(411, {"error": {"message": str(e)}})
                    return observation

            headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in HOP_BY_HOP and k.lower() != "host"
            }
            conn = http.client.HTTPConnection("127.0.0.1", lane.port, timeout=None)
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                observation.status = resp.status
                self.send_response(resp.status, resp.reason)
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                has_length = False
                for k, v in resp.getheaders():
                    lk = k.lower()
                    if lk in HOP_BY_HOP:
                        continue
                    if lk == "content-length":
                        has_length = True
                    self.send_header(k, v)
                if not has_length:
                    self.send_header("connection", "close")
                    self.close_connection = True
                self.end_headers()
                if self.command != "HEAD":
                    while True:
                        chunk = resp.read1(64 * 1024)
                        if not chunk:
                            break
                        if observation.first_byte_mono_ns is None:
                            observation.first_byte_mono_ns = time.monotonic_ns()
                            observation.first_byte_unix_ns = time.time_ns()
                        observation.response_bytes += len(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                observation.completion_reason = "completed"
            except (BrokenPipeError, ConnectionResetError) as e:
                observation.completion_reason = "client_disconnect"
                observation.error = type(e).__name__
                self.close_connection = True
            except Exception as e:
                observation.completion_reason = "proxy_error"
                observation.error = f"{type(e).__name__}: {e}"
                if not self.wfile.closed:
                    self.close_connection = True
                    print(f"[strata-multigpu] lane {lane.index} proxy error: {e}", flush=True)
            finally:
                conn.close()
            return observation

        def _dispatch(self):
            path = self.path.split("?", 1)[0]
            if path == "/__multigpu/status":
                return self._status()
            if path == "/slots" and self.command in ("GET", "HEAD"):
                return self._slots()
            if reject_generate_proxy and path in GENERATE_PATHS and self.command == "POST":
                return self._json(503, {"error": {"message": "benchmark isolation: public generation disabled"}})
            leased = path in GENERATE_PATHS and self.command == "POST"
            body = None
            requires_vision = False
            affinity_key = None
            prompt_signature: tuple[tuple[str, int], ...] = ()
            streaming = False
            request_id = None
            run_id = None
            submit_rank = None
            workload = None
            cache_state = None
            interference_arm = None
            input_tokens = None
            reusable_prefix_tokens = None
            new_prefill_tokens = None
            output_target_tokens = None
            queue_enter_mono_ns = None
            queue_enter_unix_ns = None
            admitted_mono_ns = None
            admitted_unix_ns = None
            lane_start_mono_ns = None
            lane_start_unix_ns = None
            admission_rank = None
            queue_wait_ms = None
            decision: dict = {}
            proxy_observation = ProxyObservation(completion_reason="not_started")
            if leased:
                try:
                    body = self._body()
                except ValueError as e:
                    return self._json(411, {"error": {"message": str(e)}})
                requires_vision = request_has_images(body)
                affinity_key = request_affinity_key(body, self.headers)
                prompt_signature = request_prompt_signature(body)
                streaming = request_is_streaming(body)
                request_id = next_request_id(self.headers.get("X-Strata-Benchmark-Request-Id"))
                run_id = self.headers.get("X-Strata-Benchmark-Run-Id")
                submit_rank = self.headers.get("X-Strata-Benchmark-Submit-Rank")
                workload = self.headers.get("X-Strata-Benchmark-Workload")
                cache_state = self.headers.get("X-Strata-Benchmark-Cache-State")
                interference_arm = self.headers.get("X-Strata-Benchmark-Interference-Arm")
                input_tokens = benchmark_header_int(self.headers, "X-Strata-Benchmark-Input-Tokens")
                reusable_prefix_tokens = benchmark_header_int(
                    self.headers, "X-Strata-Benchmark-Reusable-Prefix-Tokens"
                )
                new_prefill_tokens = benchmark_header_int(
                    self.headers, "X-Strata-Benchmark-New-Prefill-Tokens"
                )
                output_target_tokens = benchmark_header_int(
                    self.headers, "X-Strata-Benchmark-Output-Target-Tokens"
                )
                queue_enter_mono_ns = time.monotonic_ns()
                queue_enter_unix_ns = time.time_ns()
            try:
                if leased:
                    lane = pool.acquire(
                        requires_vision=requires_vision,
                        affinity_key=affinity_key,
                        request_bytes=len(body or b""),
                        prompt_signature=prompt_signature,
                        decision_out=decision,
                    )
                    admitted_mono_ns = time.monotonic_ns()
                    admitted_unix_ns = time.time_ns()
                    admission_rank = next_admission_rank(run_id)
                    queue_wait_ms = (admitted_mono_ns - queue_enter_mono_ns) / 1e6
                elif path in VISION_METADATA_PATHS and self.command in ("GET", "HEAD"):
                    lane = pool.metadata_lane(lane0)
                else:
                    lane = lane0
            except RuntimeError as e:
                return self._json(503, {"error": {"message": str(e)}})
            try:
                if not lane_engine_alive(lane):
                    restart_affinity = leased and decision.get("selected_reason") == "session_affinity_restart"
                    if not (restart_affinity and lane_service_ready(lane)):
                        proxy_observation = ProxyObservation(
                            status=503,
                            completion_reason="lane_unavailable",
                            error=f"GPU lane {lane.index} is not running",
                        )
                        return self._json(503, {"error": {"message": proxy_observation.error}})
                extra_headers = None
                if leased:
                    extra_headers = {
                        "X-Strata-Lane-Index": str(lane.index),
                        "X-Strata-Admission-Rank": str(admission_rank),
                        "X-Strata-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
                        "X-Strata-Benchmark-Request-Id": str(request_id),
                    }
                    lane_start_mono_ns = time.monotonic_ns()
                    lane_start_unix_ns = time.time_ns()
                proxy_observation = self._proxy(lane, body=body, extra_headers=extra_headers)
            finally:
                if leased:
                    released_mono_ns = time.monotonic_ns()
                    released_unix_ns = time.time_ns()
                    pool.release(
                        lane,
                        affinity_key=affinity_key,
                        request_bytes=len(body or b""),
                        prompt_signature=prompt_signature,
                    )
                    if bench_trace is not None or bench_console_summary:
                        first_byte_ms = (
                            (proxy_observation.first_byte_mono_ns - queue_enter_mono_ns) / 1e6
                            if proxy_observation.first_byte_mono_ns is not None else None
                        )
                        lane_first_byte_ms = (
                            (proxy_observation.first_byte_mono_ns - lane_start_mono_ns) / 1e6
                            if proxy_observation.first_byte_mono_ns is not None
                            and lane_start_mono_ns is not None else None
                        )
                        trace_record = {
                            "kind": "lane_lease",
                            "trace_schema": 2,
                            "run_id": run_id,
                            "request_id": request_id,
                            "submit_rank": submit_rank,
                            "admission_rank": admission_rank,
                            "lane_index": lane.index,
                            "gpu": lane.gpu,
                            "queue_enter_unix_ns": queue_enter_unix_ns,
                            "admitted_unix_ns": admitted_unix_ns,
                            "lane_start_unix_ns": lane_start_unix_ns,
                            "response_first_byte_unix_ns": proxy_observation.first_byte_unix_ns,
                            "released_unix_ns": released_unix_ns,
                            "queue_wait_ms": queue_wait_ms,
                            "service_ms": (released_mono_ns - admitted_mono_ns) / 1e6,
                            "e2e_ms": (released_mono_ns - queue_enter_mono_ns) / 1e6,
                            "response_first_byte_ms": first_byte_ms,
                            "lane_first_byte_ms": lane_first_byte_ms,
                            "ttft_ms": first_byte_ms if streaming else None,
                            "ttft_source": "stream_response_first_byte" if streaming else None,
                            "streaming": streaming,
                            "http_status": proxy_observation.status,
                            "response_bytes": proxy_observation.response_bytes,
                            "completion_reason": proxy_observation.completion_reason,
                            "error": proxy_observation.error,
                            "request_bytes": len(body or b""),
                            "requires_vision": requires_vision,
                            "affinity_key_prefix": affinity_key[:12] if affinity_key else None,
                            "workload": workload,
                            "cache_state": cache_state,
                            "interference_arm": interference_arm,
                            "input_tokens": input_tokens,
                            "reusable_prefix_tokens": reusable_prefix_tokens,
                            "new_prefill_tokens": new_prefill_tokens,
                            "output_target_tokens": output_target_tokens,
                            "scheduler": decision,
                        }
                        if bench_trace is not None:
                            bench_trace.write(trace_record)
                        if bench_console_summary:
                            print(benchmark_console_summary(trace_record), flush=True)

        do_GET = _dispatch
        do_HEAD = _dispatch
        do_POST = _dispatch
        do_PUT = _dispatch
        do_PATCH = _dispatch
        do_DELETE = _dispatch
        do_OPTIONS = _dispatch

    return Handler


def main() -> int:
    def _shutdown_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _shutdown_signal)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="existing Strata JSON config written by setup.py")
    ap.add_argument("--gpus", default="0,1,2", help="physical GPU ids, default: 0,1,2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18086, help="public proxy port")
    ap.add_argument("--base-port", type=int, default=19086, help="first private lane port")
    ap.add_argument("--lane-contexts", help="fixed per-lane capacities, e.g. 262144,131072,131072")
    ap.add_argument("--kv-budget", type=int, help="total host-KV token capacity guard, e.g. 524288")
    ap.add_argument("--allow-context-over-262k", action="store_true",
                    help="allow a lane above the model's currently validated 262144-token context")
    ap.add_argument("--arena-file", help="shared expert arena backing file (Linux only)")
    ap.add_argument("--state-dir", help="directory for generated lane configs")
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--cpu-partition", choices=("none", "auto"), default="auto",
                    help="partition physical CPU cores between lanes before Strata starts (default: auto)")
    ap.add_argument("--lane-cpu-cores",
                    help="exact physical-core counts per lane, e.g. 5,6,5; overrides --cpu-partition auto")
    ap.add_argument("--lane-pcie-fracs",
                    help="per-lane --pcie-frac values, e.g. 0.55,0.30,0.55")
    ap.add_argument("--lane-kv-residents",
                    help="per-lane --kv-resident token counts, e.g. 65536,32768,32768")
    ap.add_argument("--lane-vram-reserve-mibs",
                    help="per-lane --vram-reserve-mib values, e.g. 1200,1200,1200")
    ap.add_argument("--vision-lanes",
                    help="0-based lane indices allowed to serve images, e.g. 1 or 0,2; defaults to lane 0 when vision is configured")
    ap.add_argument("--private-arena", action="store_true",
                    help="benchmark only: use the normal private expert arena in each lane")
    ap.add_argument("--reject-generate-proxy", action="store_true",
                    help="benchmark only: reject generation on the supervisor proxy; private lanes still work")
    ap.add_argument("--bench-trace-jsonl",
                    help="benchmark only: append exact lane lease timing records as JSONL")
    ap.add_argument("--bench-console-summary", action="store_true",
                    help="print concise per-request benchmark summaries to stdout without persistent trace storage")
    ap.add_argument("--experimental-conversation-cache-mib", type=int, default=0,
                    help="experiment only: per-lane host-RAM MiB budget for same-lane conversation parking")
    ap.add_argument("--experimental-conversation-cache-slots", type=int, default=4,
                    help="experiment only: per-lane parked-conversation slot cap (default: 4)")
    ap.add_argument("--experimental-conversation-cache-min-free-mib", type=int, default=8192,
                    help="experiment only: host MemAvailable floor before parking (default: 8192)")
    ap.add_argument(
        "--bench-scheduler-policy",
        choices=SCHEDULER_POLICIES,
        default=SCHEDULER_POLICY_DEFAULT,
        help=(
            "select new-session placement among eligible idle lanes; "
            "the promoted balanced-additive default and safe rollback may run without trace, "
            "experimental policies require --bench-trace-jsonl"
        ),
    )
    a = ap.parse_args()

    if a.bench_scheduler_policy not in SCHEDULER_POLICIES_WITHOUT_TRACE and not a.bench_trace_jsonl:
        ap.error("experimental --bench-scheduler-policy requires --bench-trace-jsonl")
    if a.experimental_conversation_cache_mib < 0 or a.experimental_conversation_cache_slots < 1 or a.experimental_conversation_cache_min_free_mib < 0:
        ap.error("experimental conversation-cache values must be non-negative and slots must be positive")
    if os.name == "nt":
        ap.error("the shared expert arena v1 is Linux-only")
    config = Path(a.config).expanduser().resolve()
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg.get("args"), list):
        ap.error("config has no args list")

    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        ap.error("--gpus must contain unique GPU ids")
    try:
        if a.vision_lanes is not None:
            vision_lanes = parse_lane_indices(a.vision_lanes, len(gpus), what="--vision-lanes")
            if vision_lanes and not cfg.get("vision"):
                raise ValueError("--vision-lanes requires a base config with a vision entry")
        else:
            vision_lanes = {0} if cfg.get("vision") else set()
    except ValueError as e:
        ap.error(str(e))
    base_ctx_raw = option_value(cfg["args"], "--max-context")
    if base_ctx_raw is None:
        ap.error("base config has no --max-context")
    try:
        base_context = int(base_ctx_raw)
        contexts = lane_contexts(base_context, len(gpus), a.lane_contexts, a.kv_budget)
    except ValueError as e:
        ap.error(str(e))
    if not a.allow_context_over_262k and any(x > 262144 for x in contexts):
        ap.error("a lane exceeds 262144 tokens; use smaller lane capacities or --allow-context-over-262k")

    pack_raw = option_value(cfg["args"], "--pack")
    if not pack_raw:
        ap.error("base config has no --pack")
    pack = resolve_config_path(pack_raw, cfg.get("cwd"))
    try:
        spec = native_arena_spec(pack)
    except (OSError, ValueError) as e:
        ap.error(f"cannot derive the native expert arena from {pack}: {e}")

    state_dir = Path(a.state_dir).expanduser().resolve() if a.state_dir else default_state_dir(config, gpus)
    state_dir.mkdir(parents=True, exist_ok=True)
    arena_file_explicit = bool(a.arena_file)
    arena_file = Path(a.arena_file).expanduser().resolve() if arena_file_explicit else default_arena_file(pack, spec)
    arena_file.parent.mkdir(parents=True, exist_ok=True)
    arena_lease = None
    if not a.private_arena:
        try:
            arena_lease = SharedArenaLease(arena_file)
        except RuntimeError as e:
            ap.error(str(e))
    bench_trace = BenchmarkTrace(Path(a.bench_trace_jsonl).expanduser().resolve()) if a.bench_trace_jsonl else None

    try:
        require_ports_free("127.0.0.1", [a.base_port + i for i in range(len(gpus))])
        require_ports_free(a.host, [a.port])
    except RuntimeError as e:
        ap.error(str(e))

    core_groups = physical_core_cpu_sets()
    cpu_sets: list[tuple[int, ...] | None] = [None] * len(gpus)
    try:
        if a.lane_cpu_cores:
            cpu_counts = parse_int_list(a.lane_cpu_cores, what="--lane-cpu-cores")
            if len(cpu_counts) != len(gpus):
                raise ValueError(f"--lane-cpu-cores has {len(cpu_counts)} values for {len(gpus)} GPU lanes")
            cpu_sets = list(partition_cpu_sets_exact(core_groups, cpu_counts))
        elif a.cpu_partition == "auto":
            cpu_sets = list(partition_cpu_sets(core_groups, len(gpus)))
    except ValueError as e:
        ap.error(str(e))

    pcie_fracs: list[float | None] = [None] * len(gpus)
    if a.lane_pcie_fracs:
        try:
            values = parse_float_list(a.lane_pcie_fracs, what="--lane-pcie-fracs")
            if len(values) != len(gpus):
                raise ValueError(f"--lane-pcie-fracs has {len(values)} values for {len(gpus)} GPU lanes")
            pcie_fracs = list(values)
        except ValueError as e:
            ap.error(str(e))

    kv_residents: list[int | None] = [None] * len(gpus)
    if a.lane_kv_residents:
        try:
            values = parse_int_list(a.lane_kv_residents, what="--lane-kv-residents")
            if len(values) != len(gpus):
                raise ValueError(f"--lane-kv-residents has {len(values)} values for {len(gpus)} GPU lanes")
            if any(v > ctx for v, ctx in zip(values, contexts)):
                raise ValueError("--lane-kv-residents cannot exceed the corresponding lane context")
            kv_residents = list(values)
        except ValueError as e:
            ap.error(str(e))

    vram_reserves: list[int | None] = [None] * len(gpus)
    if a.lane_vram_reserve_mibs:
        try:
            values = parse_int_list(a.lane_vram_reserve_mibs, what="--lane-vram-reserve-mibs")
            if len(values) != len(gpus):
                raise ValueError(f"--lane-vram-reserve-mibs has {len(values)} values for {len(gpus)} GPU lanes")
            vram_reserves = list(values)
        except ValueError as e:
            ap.error(str(e))

    lanes: list[Lane] = []
    for i, (gpu, ctx) in enumerate(zip(gpus, contexts)):
        lane_cfg = bind_lane_gpu(
            sanitize_lane_config(
                cfg,
                allow_profile_persistence=(len(gpus) == 1),
                conversation_cache_mib=a.experimental_conversation_cache_mib,
                conversation_cache_slots=a.experimental_conversation_cache_slots,
                conversation_cache_min_free_mib=a.experimental_conversation_cache_min_free_mib,
            ),
            gpu,
        )
        lane_vision = i in vision_lanes
        lane_cfg = apply_vision_capability(lane_cfg, lane_vision)
        lane_cfg = apply_shared_arena(
            lane_cfg,
            None if a.private_arena else arena_file,
            follower=(not a.private_arena and i > 0),
        )
        lane_cfg["args"] = replace_option(lane_cfg["args"], "--max-context", ctx)
        if pcie_fracs[i] is not None:
            lane_cfg["args"] = replace_option(lane_cfg["args"], "--pcie-frac", pcie_fracs[i])
        if kv_residents[i] is not None:
            lane_cfg["args"] = replace_option(lane_cfg["args"], "--kv-resident", kv_residents[i])
        if vram_reserves[i] is not None:
            lane_cfg["args"] = replace_option(lane_cfg["args"], "--vram-reserve-mib", vram_reserves[i])
        lane_cfg["host"] = "127.0.0.1"
        if cfg.get("log"):
            log = resolve_config_path(cfg["log"], cfg.get("cwd"))
            lane_cfg["log"] = str(log.with_name(f"{log.stem}.gpu{gpu}{log.suffix or '.log'}"))
        else:
            lane_cfg["log"] = str(state_dir / f"lane-{i}-gpu{gpu}.log")
        lane_config = state_dir / f"lane-{i}-gpu{gpu}.json"
        lane_config.write_text(json.dumps(lane_cfg, indent=1), encoding="utf-8")
        lanes.append(Lane(i, gpu, a.base_port + i, ctx, lane_config, cpu_sets[i], pcie_fracs[i], kv_residents[i],
                          lane_vision, vram_reserves[i]))

    print(
        f"[strata-multigpu] {len(lanes)} lanes; context capacities {contexts} "
        f"(total {sum(contexts):,}); shared expert arena {spec.expert_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    print(f"[strata-multigpu] arena backing: {arena_file} ({spec.bytes:,} bytes)", flush=True)
    if bench_trace is not None:
        bench_trace.write({
            "kind": "benchmark_manifest",
            "trace_schema": 2,
            "created_unix_ns": time.time_ns(),
            "fork_commit": repository_head(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "kernel": os.uname().release if hasattr(os, "uname") else None,
            "config_path": str(config),
            "config_sha256": file_sha256(config),
            "pack_path": str(pack),
            "shared_arena": not a.private_arena,
            "arena_file": str(arena_file) if not a.private_arena else None,
            "arena_bytes": spec.bytes,
            "kv_budget": a.kv_budget,
            "scheduler_policy": a.bench_scheduler_policy,
            "scheduler_policy_scope": "new_session_idle_lane_only",
            "conversation_parking": {"mib_per_lane": a.experimental_conversation_cache_mib, "slots_per_lane": a.experimental_conversation_cache_slots, "min_free_mib": a.experimental_conversation_cache_min_free_mib},
            "public_host": a.host,
            "public_port": a.port,
            "base_port": a.base_port,
            "lanes": [
                {
                    "index": lane.index,
                    "gpu": lane.gpu,
                    "port": lane.port,
                    "context": lane.context,
                    "cpus": list(lane.cpus) if lane.cpus else None,
                    "pcie_frac": lane.pcie_frac,
                    "kv_resident": lane.kv_resident,
                    "vision": lane.vision,
                    "vram_reserve_mib": lane.vram_reserve_mib,
                    "config_path": str(lane.config),
                    "config_sha256": file_sha256(lane.config),
                }
                for lane in lanes
            ],
        })
        print(f"[strata-multigpu] benchmark lease trace: {bench_trace.path}", flush=True)
    if vision_lanes:
        print(
            "[strata-multigpu] vision lanes: " +
            ", ".join(f"lane {x.index}=GPU{x.gpu}" for x in lanes if x.vision),
            flush=True,
        )
    if any(x.cpus for x in lanes):
        print(
            "[strata-multigpu] CPU partitions: " +
            "; ".join(f"lane {x.index}={','.join(map(str, x.cpus or ())) }" for x in lanes),
            flush=True,
        )
    if any(x.pcie_frac is not None for x in lanes):
        print(
            "[strata-multigpu] PCIe fractions: " +
            ", ".join(f"lane {x.index}={x.pcie_frac:.3f}" for x in lanes if x.pcie_frac is not None),
            flush=True,
        )
    if any(x.kv_resident is not None for x in lanes):
        print(
            "[strata-multigpu] KV resident: " +
            ", ".join(f"lane {x.index}={x.kv_resident}" for x in lanes if x.kv_resident is not None),
            flush=True,
        )
    if a.experimental_conversation_cache_mib:
        print(f"[strata-multigpu] EXPERIMENTAL same-lane conversation parking: {a.experimental_conversation_cache_mib} MiB/lane, {a.experimental_conversation_cache_slots} slots/lane, min-free {a.experimental_conversation_cache_min_free_mib} MiB", flush=True)
    if any(x.vram_reserve_mib is not None for x in lanes):
        print(
            "[strata-multigpu] VRAM reserve MiB: " +
            ", ".join(f"lane {x.index}={x.vram_reserve_mib}" for x in lanes if x.vram_reserve_mib is not None),
            flush=True,
        )

    started: list[Lane] = []
    try:
        for lane in lanes:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = lane.gpu
            # Pre-0.1.30 fork builds used these variables to intercept mmap.
            # Upstream 0.1.30 owns the feature through --shared-expert-arena;
            # strip inherited legacy knobs so lane configs are the sole source.
            env.pop("STRATA_SHARED_ARENA_FILE", None)
            env.pop("STRATA_SHARED_ARENA_BYTES", None)
            cmd = [
                sys.executable, str(ROOT / "serve" / "server.py"),
                "--engine", "strata",
                "--config", str(lane.config),
                "--host", "127.0.0.1",
                "--port", str(lane.port),
            ]
            print(
                f"[strata-multigpu] starting lane {lane.index}: GPU {lane.gpu}, "
                f"context {lane.context:,}, port {lane.port}",
                flush=True,
            )
            preexec_fn = None
            if lane.cpus and hasattr(os, "sched_setaffinity"):
                preexec_fn = lambda cpus=lane.cpus: _set_process_affinity(cpus)
            lane.process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, preexec_fn=preexec_fn)
            started.append(lane)
            wait_ready(lane, a.startup_timeout)
            print(f"[strata-multigpu] lane {lane.index} ready", flush=True)

        pool = LanePool(lanes, scheduler_policy=a.bench_scheduler_policy)
        httpd = ThreadingHTTPServer(
            (a.host, a.port),
            make_handler(
                pool,
                lanes[0],
                arena_file,
                spec.bytes,
                a.kv_budget,
                a.reject_generate_proxy,
                bench_trace,
                a.bench_console_summary,
            ),
        )
        print(
            f"[strata-multigpu] ready: http://{a.host}:{a.port}/v1 "
            f"({len(lanes)} concurrent generation lanes)",
            flush=True,
        )
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
    except KeyboardInterrupt:
        pass
    finally:
        for lane in reversed(started):
            stop_lane(lane)
        if arena_lease is not None:
            arena_lease.close()
        if not a.private_arena and not arena_file_explicit:
            try:
                arena_file.unlink(missing_ok=True)
                Path(str(arena_file) + ".lock").unlink(missing_ok=True)
            except OSError as e:
                print(f"[strata-multigpu] warning: could not remove managed shared arena: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
