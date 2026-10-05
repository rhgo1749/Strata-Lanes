from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("strata_multigpu_server", HERE / "multigpu_server.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class _AliveProcess:
    pid = 12345

    def poll(self):
        return None


class MultiGpuPlanningTests(unittest.TestCase):
    def setUp(self):
        self._lane_engine_alive = M.lane_engine_alive
        M.lane_engine_alive = lambda lane: lane.process is not None and lane.process.poll() is None

    def tearDown(self):
        M.lane_engine_alive = self._lane_engine_alive

    def test_replace_existing_option(self):
        args = ["--pack", "/m", "--max-context", "131072", "--kv", "int8"]
        self.assertEqual(
            M.replace_option(args, "--max-context", 262144),
            ["--pack", "/m", "--max-context", "262144", "--kv", "int8"],
        )
        self.assertEqual(args[3], "131072")

    def test_replace_missing_option(self):
        self.assertEqual(M.replace_option(["--pack", "/m"], "--max-context", 65536),
                         ["--pack", "/m", "--max-context", "65536"])

    def test_remove_option_removes_all_layer_split_pairs(self):
        self.assertEqual(
            M.remove_option(["--pack", "/m", "--layer-split", "auto", "--kv", "int8",
                             "--layer-split", "16,32"], "--layer-split"),
            ["--pack", "/m", "--kv", "int8"],
        )

    def test_lane_config_cannot_reexpand_into_layer_split(self):
        cfg = {"gpu": [0, 1, 2], "layer_split": "16,32",
               "args": ["--pack", "/m", "--layer-split", "auto", "--max-context", "262144"]}
        got = M.sanitize_lane_config(cfg)
        self.assertNotIn("gpu", got)
        self.assertNotIn("layer_split", got)
        self.assertNotIn("--layer-split", got["args"])
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-mib"), "0")
        self.assertIn("gpu", cfg)
        self.assertIn("--layer-split", cfg["args"])

    def test_lane_config_strips_upstream_multigpu_and_shared_profile_writers(self):
        cfg = {
            "expert_profile_save": "/tmp/top-level-shared.profile",
            "expert_profile_save_every": 7,
            "args": [
            "--pack", "/m",
            "--split-device", "1",
            "--peer-device", "1",
            "--peer-reserve-mib", "700",
            "--peer-slots", "1024",
            "--peer-adapt-swaps", "2",
            "--peer-prefill-rows", "4096",
            "--expert-cache-device1", "100",
            "--expert-cache-device2", "200",
            "--expert-cache-device3", "300",
            "--expert-cache-remote-placement", "layer",
            "--split-skip-if-fits",
            "--expert-profile-save", "/tmp/shared.profile",
            "--expert-profile-save-every", "5",
            "--expert-profile", "/tmp/read-only.profile",
        ]}
        got = M.sanitize_lane_config(cfg)
        for name in (
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
            "--split-skip-if-fits",
            "--expert-profile-save",
            "--expert-profile-save-every",
        ):
            self.assertNotIn(name, got["args"])
        self.assertEqual(M.option_value(got["args"], "--expert-profile"), "/tmp/read-only.profile")
        self.assertNotIn("expert_profile_save", got)
        self.assertNotIn("expert_profile_save_every", got)

    def test_single_lane_sanitizer_preserves_profile_persistence(self):
        cfg = {
            "expert_profile_save": "/tmp/learned.profile",
            "expert_profile_save_every": 7,
            "args": [
                "--pack", "/m",
                "--expert-profile-save", "/tmp/learned-cli.profile",
                "--expert-profile-save-every", "5",
                "--expert-profile", "/tmp/read-only.profile",
            ],
        }
        got = M.sanitize_lane_config(cfg, allow_profile_persistence=True)
        self.assertEqual(got["expert_profile_save"], "/tmp/learned.profile")
        self.assertEqual(got["expert_profile_save_every"], 7)
        self.assertEqual(M.option_value(got["args"], "--expert-profile-save"), "/tmp/learned-cli.profile")
        self.assertEqual(M.option_value(got["args"], "--expert-profile-save-every"), "5")
        self.assertEqual(M.option_value(got["args"], "--expert-profile"), "/tmp/read-only.profile")

    def test_bound_lane_config_uses_single_physical_gpu_for_telemetry(self):
        cfg = {"gpu": [0, 1, 2], "layer_split": "16,32",
               "args": ["--pack", "/m", "--layer-split", "auto"]}
        got = M.bind_lane_gpu(M.sanitize_lane_config(cfg), "2")
        self.assertEqual(got["gpu"], 2)
        self.assertNotIn("layer_split", got)
        self.assertNotIn("--layer-split", got["args"])

    def test_lane_config_strips_upstream_internal_parallelism(self):
        cfg = {
            "parallel": 4,
            "args": [
                "--pack", "/m",
                "--batch", "4",
                "--slots", "3",
                "--batch-groups", "2",
                "--trim-stage-weights",
            ],
        }
        got = M.sanitize_lane_config(cfg)
        self.assertNotIn("parallel", got)
        self.assertNotIn("--batch", got["args"])
        self.assertNotIn("--slots", got["args"])
        self.assertNotIn("--batch-groups", got["args"])
        self.assertNotIn("--trim-stage-weights", got["args"])

    def test_responses_api_is_generation_work(self):
        self.assertIn("/v1/responses", M.GENERATE_PATHS)

    def test_lane_config_disables_upstream_conversation_parking(self):
        cfg = {"args": ["--pack", "/m", "--conversation-cache-mib", "8192",
                         "--conversation-cache-slots", "4"]}
        got = M.sanitize_lane_config(cfg)
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-mib"), "0")
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-slots"), "4")

    def test_lane_config_can_enable_experimental_same_lane_parking(self):
        cfg = {"args": ["--pack", "/m", "--conversation-cache-mib", "0"]}
        got = M.sanitize_lane_config(cfg, conversation_cache_mib=4096,
                                     conversation_cache_slots=4, conversation_cache_min_free_mib=8192)
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-mib"), "4096")
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-slots"), "4")
        self.assertEqual(M.option_value(got["args"], "--conversation-cache-min-free-mib"), "8192")

    def test_nonvision_lane_drops_encoder_and_vram_reserve(self):
        cfg = {"vision": {"exe": "/v", "gpu": True},
               "args": ["--pack", "/m", "--vision", "--vram-reserve-mib", "700", "--max-context", "262144"]}
        got = M.apply_vision_capability(M.sanitize_lane_config(cfg), False)
        self.assertNotIn("vision", got)
        self.assertNotIn("--vision", got["args"])
        self.assertNotIn("--vram-reserve-mib", got["args"])
        self.assertIn("vision", cfg)

    def test_vision_lane_keeps_encoder_config(self):
        cfg = {"vision": {"exe": "/v", "gpu": True},
               "args": ["--pack", "/m", "--vision", "--vram-reserve-mib", "700"]}
        got = M.apply_vision_capability(M.sanitize_lane_config(cfg), True)
        self.assertIn("vision", got)
        self.assertIn("--vision", got["args"])
        self.assertIn("--vram-reserve-mib", got["args"])

    def test_nonvision_lane_can_reapply_explicit_vram_reserve(self):
        cfg = {"vision": {"exe": "/v", "gpu": True},
               "args": ["--pack", "/m", "--vision", "--vram-reserve-mib", "700"]}
        got = M.apply_vision_capability(M.sanitize_lane_config(cfg), False)
        self.assertNotIn("--vram-reserve-mib", got["args"])
        got["args"] = M.replace_option(got["args"], "--vram-reserve-mib", 1200)
        self.assertEqual(M.option_value(got["args"], "--vram-reserve-mib"), "1200")

    def test_shared_arena_uses_upstream_cli_and_replaces_stale_value(self):
        cfg = {"args": ["--pack", "/m", "--shared-expert-arena", "/old"]}
        got = M.apply_shared_arena(cfg, Path("/dev/shm/strata/new.bin"))
        self.assertEqual(M.option_value(got["args"], "--shared-expert-arena"), "/dev/shm/strata/new.bin")
        self.assertEqual(got["args"].count("--shared-expert-arena"), 1)
        self.assertNotIn("--shared-expert-arena-follower", got["args"])
        self.assertEqual(M.option_value(cfg["args"], "--shared-expert-arena"), "/old")

    def test_shared_arena_follower_is_explicit_and_replaces_stale_flag(self):
        cfg = {"args": ["--pack", "/m", "--shared-expert-arena", "/old",
                        "--shared-expert-arena-follower"]}
        leader = M.apply_shared_arena(cfg, Path("/dev/shm/strata/new.bin"))
        follower = M.apply_shared_arena(cfg, Path("/dev/shm/strata/new.bin"), follower=True)
        self.assertNotIn("--shared-expert-arena-follower", leader["args"])
        self.assertEqual(follower["args"].count("--shared-expert-arena-follower"), 1)

    def test_private_arena_strips_inherited_shared_arena(self):
        cfg = {"args": ["--pack", "/m", "--shared-expert-arena", "/old",
                        "--shared-expert-arena-follower", "--kv", "int8"]}
        got = M.apply_shared_arena(cfg, None)
        self.assertNotIn("--shared-expert-arena", got["args"])
        self.assertNotIn("--shared-expert-arena-follower", got["args"])
        self.assertEqual(got["args"], ["--pack", "/m", "--kv", "int8"])

    def test_default_shared_arena_is_tmpfs(self):
        spec = M.ArenaSpec(bytes=100, expert_bytes=80, max_blob=20, n_expert=512)
        path = M.default_arena_file(Path("/models/pack"), spec)
        self.assertEqual(path.parts[:3], ("/", "dev", "shm"))

    @unittest.skipIf(M.fcntl is None, "flock is Linux-only")
    def test_shared_arena_lease_refuses_a_second_supervisor(self):
        with tempfile.TemporaryDirectory() as td:
            arena = Path(td) / "arena.bin"
            first = M.SharedArenaLease(arena)
            try:
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    M.SharedArenaLease(arena)
            finally:
                first.close()
            second = M.SharedArenaLease(arena)
            second.close()

    def test_port_preflight_rejects_stale_listener(self):
        sock = M.socket.socket(M.socket.AF_INET, M.socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        try:
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                M.require_ports_free("127.0.0.1", [port])
        finally:
            sock.close()

    def test_port_preflight_accepts_free_port_and_ephemeral_public_port(self):
        sock = M.socket.socket(M.socket.AF_INET, M.socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        M.require_ports_free("127.0.0.1", [port, 0])

    def test_benchmark_trace_records_exact_lane_lease_and_headers(self):
        class Backend(M.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n:
                    self.rfile.read(n)
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        backend = M.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        lane = M.Lane(0, "0", backend.server_address[1], 32768, Path("lane.json"), process=_AliveProcess())
        pool = M.LanePool([lane])

        with tempfile.TemporaryDirectory() as td:
            trace_path = Path(td) / "leases.jsonl"
            trace = M.BenchmarkTrace(trace_path)
            handler = M.make_handler(pool, lane, Path("/dev/shm/fake.bin"), 123, None, bench_trace=trace)
            proxy = M.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            proxy_thread.start()
            try:
                conn = M.http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=2)
                body = b'{"stream":true,"messages":[{"role":"user","content":"hi"}]}'
                conn.request(
                    "POST", "/v1/chat/completions", body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                        "X-Strata-Benchmark-Run-Id": "run-a",
                        "X-Strata-Benchmark-Request-Id": "req-3",
                        "X-Strata-Benchmark-Submit-Rank": "3",
                        "X-Strata-Benchmark-Workload": "unit-chat",
                        "X-Strata-Benchmark-Cache-State": "cold",
                        "X-Strata-Benchmark-Interference-Arm": "solo",
                        "X-Strata-Benchmark-Input-Tokens": "11",
                        "X-Strata-Benchmark-Reusable-Prefix-Tokens": "0",
                        "X-Strata-Benchmark-New-Prefill-Tokens": "11",
                        "X-Strata-Benchmark-Output-Target-Tokens": "8",
                    },
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.getheader("X-Strata-Lane-Index"), "0")
                self.assertEqual(resp.getheader("X-Strata-Admission-Rank"), "0")
                self.assertEqual(resp.getheader("X-Strata-Benchmark-Request-Id"), "req-3")
                self.assertGreaterEqual(float(resp.getheader("X-Strata-Queue-Wait-Ms")), 0.0)
                self.assertEqual(resp.read(), b'{"ok":true}')

                conn.request(
                    "POST", "/v1/chat/completions", body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                        "X-Strata-Benchmark-Run-Id": "run-b",
                        "X-Strata-Benchmark-Request-Id": "req-b0",
                        "X-Strata-Benchmark-Submit-Rank": "0",
                    },
                )
                resp2 = conn.getresponse()
                self.assertEqual(resp2.status, 200)
                self.assertEqual(resp2.getheader("X-Strata-Admission-Rank"), "0")
                self.assertEqual(resp2.read(), b'{"ok":true}')
                conn.close()
            finally:
                proxy.shutdown()
                proxy.server_close()
                backend.shutdown()
                backend.server_close()
                proxy_thread.join(1.0)
                backend_thread.join(1.0)

            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            rec = records[0]
            self.assertEqual(rec["run_id"], "run-a")
            self.assertEqual(rec["request_id"], "req-3")
            self.assertEqual(rec["submit_rank"], "3")
            self.assertEqual(rec["admission_rank"], 0)
            self.assertEqual(rec["lane_index"], 0)
            self.assertGreaterEqual(rec["queue_wait_ms"], 0.0)
            self.assertGreaterEqual(rec["service_ms"], 0.0)
            self.assertGreaterEqual(rec["e2e_ms"], rec["service_ms"])
            self.assertGreaterEqual(rec["response_first_byte_ms"], 0.0)
            self.assertEqual(rec["ttft_ms"], rec["response_first_byte_ms"])
            self.assertEqual(rec["completion_reason"], "completed")
            self.assertEqual(rec["http_status"], 200)
            self.assertEqual(rec["response_bytes"], len(b'{"ok":true}'))
            self.assertEqual(rec["workload"], "unit-chat")
            self.assertEqual(rec["cache_state"], "cold")
            self.assertEqual(rec["interference_arm"], "solo")
            self.assertEqual(rec["input_tokens"], 11)
            self.assertEqual(rec["reusable_prefix_tokens"], 0)
            self.assertEqual(rec["new_prefill_tokens"], 11)
            self.assertEqual(rec["output_target_tokens"], 8)
            self.assertEqual(rec["scheduler"]["policy"], M.SCHEDULER_POLICY_DEFAULT)
            self.assertEqual(rec["scheduler"]["selected_reason"], "session_start_balance_additive")
            self.assertEqual(rec["scheduler"]["session_turn"], 1)
            self.assertEqual(rec["scheduler"]["queued_request_count"], 0)
            self.assertEqual(rec["scheduler"]["lane_components"][0]["lane_index"], 0)
            self.assertEqual(rec["scheduler"]["lane_components"][0]["estimated_reusable_prefix_bytes"], 0)
            self.assertLessEqual(rec["queue_enter_unix_ns"], rec["admitted_unix_ns"])
            self.assertLessEqual(rec["admitted_unix_ns"], rec["lane_start_unix_ns"])
            self.assertLessEqual(rec["lane_start_unix_ns"], rec["response_first_byte_unix_ns"])
            self.assertLessEqual(rec["response_first_byte_unix_ns"], rec["released_unix_ns"])
            self.assertEqual(records[1]["run_id"], "run-b")
            self.assertEqual(records[1]["admission_rank"], 0)
            self.assertEqual(records[1]["scheduler"]["selected_reason"], "session_affinity")
            self.assertEqual(records[1]["scheduler"]["session_turn"], 2)
            self.assertGreater(
                records[1]["scheduler"]["lane_components"][0]["estimated_reusable_prefix_bytes"], 0
            )

    def test_console_summary_can_run_without_jsonl_trace(self):
        class Backend(M.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n:
                    self.rfile.read(n)
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        backend = M.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
        backend_thread.start()
        lane = M.Lane(0, "0", backend.server_address[1], 32768, Path("lane.json"), process=_AliveProcess())
        pool = M.LanePool([lane])
        handler = M.make_handler(
            pool,
            lane,
            Path("/dev/shm/fake.bin"),
            123,
            None,
            bench_console_summary=True,
        )
        proxy = M.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                conn = M.http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=2)
                body = b'{"stream":true,"messages":[{"role":"user","content":"hi"}]}'
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body=body,
                    headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
                )
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.read(), b'{"ok":true}')
                deadline = time.monotonic() + 1.0
                while "[strata-multigpu][bench]" not in output.getvalue() and time.monotonic() < deadline:
                    time.sleep(0.005)
                conn.close()
        finally:
            proxy.shutdown()
            proxy.server_close()
            backend.shutdown()
            backend.server_close()
            proxy_thread.join(1.0)
            backend_thread.join(1.0)
        rendered = output.getvalue()
        self.assertIn("[strata-multigpu][bench]", rendered)
        self.assertIn("lane=0", rendered)

    def test_benchmark_console_summary_is_concise_and_privacy_safe(self):
        record = {
            "lane_index": 2,
            "queue_wait_ms": 12.345,
            "ttft_ms": 98.765,
            "e2e_ms": 456.789,
            "http_status": 200,
            "completion_reason": "completed",
            "request_id": "secret-request-label",
            "affinity_key_prefix": "deadbeefcafe",
            "scheduler": {"selected_reason": "session_affinity"},
        }
        summary = M.benchmark_console_summary(record)
        self.assertIn("lane=2", summary)
        self.assertIn("reason=session_affinity", summary)
        self.assertIn("queue=12.3ms", summary)
        self.assertIn("ttft=98.8ms", summary)
        self.assertIn("e2e=456.8ms", summary)
        self.assertIn("status=200", summary)
        self.assertIn("result=completed", summary)
        self.assertNotIn("secret-request-label", summary)
        self.assertNotIn("deadbeefcafe", summary)

    def test_prompt_signature_estimates_only_exact_leading_messages(self):
        first = json.dumps({
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "first"},
            ]
        }).encode()
        continuation = json.dumps({
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "next"},
            ]
        }).encode()
        changed = json.dumps({
            "messages": [
                {"role": "system", "content": "different"},
                {"role": "user", "content": "first"},
            ]
        }).encode()
        base = M.request_prompt_signature(first)
        later = M.request_prompt_signature(continuation)
        self.assertEqual(M.reusable_prefix_bytes(later, base), sum(size for _hash, size in base))
        self.assertEqual(M.reusable_prefix_bytes(M.request_prompt_signature(changed), base), 0)
        self.assertNotIn("rules", repr(base))
        self.assertTrue(M.request_is_streaming(b'{"stream":true}'))
        self.assertFalse(M.request_is_streaming(b'{"stream":false}'))

    def test_parse_vision_lane_indices(self):
        self.assertEqual(M.parse_lane_indices("1", 3, what="--vision-lanes"), {1})
        self.assertEqual(M.parse_lane_indices("0,2", 3, what="--vision-lanes"), {0, 2})
        self.assertEqual(M.parse_lane_indices("none", 3, what="--vision-lanes"), set())
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            M.parse_lane_indices("3", 3, what="--vision-lanes")

    def test_detect_supported_image_message_parts(self):
        openai = b'{"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}'
        anthropic = b'{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]}]}'
        text = b'{"messages":[{"role":"user","content":"hello"}]}'
        self.assertTrue(M.request_has_images(openai))
        self.assertTrue(M.request_has_images(anthropic))
        self.assertFalse(M.request_has_images(text))
        self.assertFalse(M.request_has_images(b"not-json"))

    def test_affinity_key_survives_appended_chat_history(self):
        first = {"messages": [{"role": "system", "content": "rules"},
                              {"role": "user", "content": "build the thing"}]}
        later = {"messages": first["messages"] + [{"role": "assistant", "content": "working"},
                                                   {"role": "user", "content": "continue"}]}
        self.assertEqual(
            M.request_affinity_key(json.dumps(first).encode()),
            M.request_affinity_key(json.dumps(later).encode()),
        )

    def test_explicit_affinity_id_wins_over_message_fallback(self):
        body = b'{"messages":[{"role":"user","content":"hello"}]}'
        explicit = M.request_affinity_key(body, {"x-strata-session-id": "session-a"})
        fallback = M.request_affinity_key(body)
        self.assertIsNotNone(explicit)
        self.assertNotEqual(explicit, fallback)
        self.assertEqual(explicit, M.request_affinity_key(b'{"messages":[]}', {"x-strata-session-id": "session-a"}))

    def test_vision_request_uses_only_vision_lane(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), vision=False, process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), vision=True, process=_AliveProcess()),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), vision=False, process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        got = pool.acquire(requires_vision=True)
        self.assertEqual(got.index, 1)
        pool.release(got)

    def test_text_request_yields_vision_lane_to_waiting_image(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), vision=False, process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), vision=True, process=_AliveProcess()),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), vision=False, process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.cursor = 1
        pool.vision_waiters = 1
        got = pool.acquire()
        self.assertEqual(got.index, 2)
        pool.release(got)

    def test_affinity_waits_for_its_busy_lane_instead_of_spilling(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(), busy=True),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.affinity["chat-a"] = 0
        pool.cursor = 1
        started = threading.Event()
        result = []

        def acquire():
            started.set()
            result.append(pool.acquire(affinity_key="chat-a"))

        thread = threading.Thread(target=acquire)
        thread.start()
        self.assertTrue(started.wait(1.0))
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        pool.release(lanes[0])
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].index, 0)
        pool.release(result[0])

    def test_affinity_waiter_reserves_released_lane_from_new_session(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(), busy=True),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(), busy=True),
        ]
        pool = M.LanePool(lanes)
        pool.affinity["returning"] = 0
        returning = []
        newcomer = []

        t_returning = threading.Thread(target=lambda: returning.append(pool.acquire(affinity_key="returning")))
        t_returning.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with pool.cv:
                if pool.affinity_waiters[0] == 1:
                    break
            time.sleep(0.005)
        self.assertEqual(pool.affinity_waiters[0], 1)

        t_new = threading.Thread(target=lambda: newcomer.append(pool.acquire(affinity_key="new-chat")))
        t_new.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with pool.cv:
                if len(pool.waiters) == 1:
                    break
            time.sleep(0.005)
        self.assertEqual(len(pool.waiters), 1)

        pool.release(lanes[0], affinity_key="old", request_bytes=1000)
        t_returning.join(1.0)
        self.assertFalse(t_returning.is_alive())
        self.assertEqual(returning[0].index, 0)
        time.sleep(0.05)
        self.assertTrue(t_new.is_alive())

        pool.release(lanes[1], affinity_key="other", request_bytes=1000)
        t_new.join(1.0)
        self.assertFalse(t_new.is_alive())
        self.assertEqual(newcomer[0].index, 1)
        pool.release(returning[0], affinity_key="returning", request_bytes=1100)
        pool.release(newcomer[0], affinity_key="new-chat", request_bytes=500)

    def test_four_waiting_new_sessions_use_compatible_fifo_order(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(), busy=True),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(), busy=True),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(), busy=True),
        ]
        pool = M.LanePool(lanes)
        results = []
        threads = []

        for i in range(4):
            thread = threading.Thread(
                target=lambda i=i: results.append((i, pool.acquire(affinity_key=f"q{i}").index))
            )
            thread.start()
            threads.append(thread)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with pool.cv:
                    if len(pool.waiters) == i + 1:
                        break
                time.sleep(0.005)
            self.assertEqual(len(pool.waiters), i + 1)

        for expected, lane_index in ((0, 2), (1, 0), (2, 1)):
            pool.release(lanes[lane_index], affinity_key=f"initial-{lane_index}", request_bytes=1000)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and len(results) <= expected:
                time.sleep(0.005)
            self.assertEqual(results[expected], (expected, lane_index))

        self.assertTrue(threads[3].is_alive())
        pool.release(lanes[2], affinity_key="q0", request_bytes=500)
        threads[3].join(1.0)
        self.assertFalse(threads[3].is_alive())
        self.assertEqual(results[3], (3, 2))
        for thread in threads[:3]:
            thread.join(1.0)
        for lane in lanes:
            if lane.busy:
                pool.release(lane)

    def test_new_session_waits_when_all_lanes_busy(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(), busy=True,
                   live_affinity_key="a", live_request_bytes=200000),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(), busy=True,
                   live_affinity_key="b", live_request_bytes=60000),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(), busy=True,
                   live_affinity_key="c", live_request_bytes=700),
        ]
        pool = M.LanePool(lanes)
        started = threading.Event()
        result = []

        def acquire():
            started.set()
            result.append(pool.acquire(affinity_key="new-chat", request_bytes=500))

        thread = threading.Thread(target=acquire)
        thread.start()
        self.assertTrue(started.wait(1.0))
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())
        pool.release(lanes[1], affinity_key="b", request_bytes=60000)
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result[0].index, 1)
        pool.release(result[0], affinity_key="new-chat", request_bytes=500)

    def test_queue_status_counts_waiting_work_without_changing_fifo(self):
        lane = M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(), busy=True)
        pool = M.LanePool([lane])
        acquired = []
        thread = threading.Thread(
            target=lambda: acquired.append(pool.acquire(affinity_key="queued", request_bytes=4321))
        )
        thread.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            status = pool.queue_status()
            if status["new_session_waiters"] == 1:
                break
            time.sleep(0.005)
        status = pool.queue_status()
        self.assertEqual(status["new_session_waiters"], 1)
        self.assertEqual(status["queued_request_bytes"], 4321)
        self.assertEqual(status["busy_lanes"], 1)
        pool.release(lane, affinity_key="initial", request_bytes=10)
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(acquired[0].index, 0)
        pool.release(acquired[0], affinity_key="queued", request_bytes=4321)

    def test_active_work_is_separate_from_retained_lane_state(self):
        lane = M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess())
        pool = M.LanePool([lane])
        signature = (("prompt-a", 123), ("prompt-b", 77))

        got = pool.acquire(
            affinity_key="session-a",
            request_bytes=4567,
            prompt_signature=signature,
        )
        self.assertIs(got, lane)
        self.assertTrue(lane.busy)
        self.assertEqual(lane.active_affinity_key, "session-a")
        self.assertEqual(lane.active_request_bytes, 4567)
        self.assertEqual(lane.active_prompt_signature, signature)
        self.assertIsNotNone(lane.active_started_mono_ns)
        self.assertEqual(lane.live_request_bytes, 0)
        self.assertEqual(lane.live_prompt_signature, ())

        pool.release(
            lane,
            affinity_key="session-a",
            request_bytes=4567,
            prompt_signature=signature,
        )
        self.assertFalse(lane.busy)
        self.assertIsNone(lane.active_affinity_key)
        self.assertEqual(lane.active_request_bytes, 0)
        self.assertEqual(lane.active_prompt_signature, ())
        self.assertIsNone(lane.active_started_mono_ns)
        self.assertEqual(lane.live_request_bytes, 4567)
        self.assertEqual(lane.live_prompt_signature, signature)

    def test_decision_record_exposes_phase2_queue_and_active_proxies(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), vision=True, process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.affinity.update({"a": 0, "b": 0, "c": 1})
        now = time.monotonic_ns()
        pool.waiters[10] = M.Waiter(False, 4000, now - 5_000_000)
        pool.waiters[11] = M.Waiter(True, 2000, now - 10_000_000)
        pool._start_lane_work(
            lanes[0],
            affinity_key="active-a",
            request_bytes=1234,
            prompt_signature=(("active", 50),),
        )

        record = pool._decision_record(
            alive=lanes,
            eligible=lanes,
            selected=lanes[1],
            selected_reason="test",
            request_bytes=999,
            prompt_signature=(("incoming", 60),),
            session_turn=1,
        )
        by_lane = {row["lane_index"]: row for row in record["lane_components"]}

        self.assertEqual(by_lane[0]["affinity_session_count"], 2)
        self.assertEqual(by_lane[1]["affinity_session_count"], 1)
        self.assertEqual(by_lane[0]["active_request_bytes"], 1234)
        self.assertEqual(by_lane[0]["active_prompt_message_bytes"], 50)
        self.assertIsNotNone(by_lane[0]["active_elapsed_ms"])
        self.assertEqual(by_lane[0]["compatible_queued_request_count"], 1)
        self.assertEqual(by_lane[0]["compatible_queued_request_bytes"], 4000)
        self.assertGreaterEqual(by_lane[0]["oldest_compatible_queue_age_ms"], 5.0)
        self.assertEqual(by_lane[1]["compatible_queued_request_count"], 2)
        self.assertEqual(by_lane[1]["compatible_queued_request_bytes"], 6000)
        self.assertGreaterEqual(by_lane[1]["oldest_compatible_queue_age_ms"], 10.0)

    def test_benchmark_scheduler_policies_choose_expected_idle_lane(self):
        incoming = (("same-prefix", 100),)
        expected = {
            M.SCHEDULER_POLICY_SAFE: 2,
            "round-robin-idle-v1": 0,
            "retained-state-smallest-v1": 2,
            "cache-aware-idle-v1": 0,
            "additive-new-prefill-retained-state-proxy-v1": 2,
            "multiplicative-new-prefill-retained-state-proxy-v1": 0,
            "session-start-balance-cache-aware-v1": 1,
            "balanced-additive-new-prefill-retained-state-proxy-v1": 1,
        }

        for policy, expected_lane in expected.items():
            with self.subTest(policy=policy):
                lanes = [
                    M.Lane(
                        0, "0", 19086, 262144, Path("lane0.json"),
                        process=_AliveProcess(),
                        live_request_bytes=9000,
                        live_sequence=1,
                        live_prompt_signature=incoming,
                    ),
                    M.Lane(
                        1, "1", 19087, 262144, Path("lane1.json"),
                        process=_AliveProcess(),
                        live_request_bytes=1000,
                        live_sequence=2,
                    ),
                    M.Lane(
                        2, "2", 19088, 262144, Path("lane2.json"),
                        process=_AliveProcess(),
                    ),
                ]
                pool = M.LanePool(lanes, scheduler_policy=policy)
                pool.affinity.update({"old-a": 0, "old-b": 0, "old-c": 2})
                decision = {}
                got = pool.acquire(
                    affinity_key="new-session",
                    request_bytes=500,
                    prompt_signature=incoming,
                    decision_out=decision,
                )
                self.assertEqual(got.index, expected_lane)
                self.assertEqual(decision["policy"], policy)
                self.assertEqual(decision["policy_scope"], "new_session_idle_lane_only")
                self.assertEqual(
                    decision["selected_reason"],
                    (
                        "live_state_lexicographic"
                        if policy == M.SCHEDULER_POLICY_SAFE
                        else (
                            "session_start_balance_additive"
                            if policy == M.SCHEDULER_POLICY_DEFAULT
                            else "benchmark_policy_score"
                        )
                    ),
                )
                selected = next(
                    row for row in decision["lane_components"]
                    if row["lane_index"] == expected_lane
                )
                self.assertEqual(decision["selected_placement_score"], selected["placement_score"])
                self.assertEqual(decision["selected_placement_key"], selected["placement_key"])
                self.assertEqual(
                    decision["load_proxy_source"],
                    "retained_last_completed_request_bytes_v1",
                )
                if policy == M.SCHEDULER_POLICY_SAFE:
                    self.assertEqual(selected["placement_score"], selected["placement_key"])
                pool.release(
                    got,
                    affinity_key="new-session",
                    request_bytes=500,
                    prompt_signature=incoming,
                )

    def test_balanced_additive_avoids_long_lived_session_start_attractor(self):
        lanes = [
            M.Lane(
                0, "0", 19086, 262144, Path("lane0.json"),
                process=_AliveProcess(),
                live_request_bytes=15335,
                live_sequence=11,
                live_prompt_signature=(("shared", 14989), ("lane0", 346)),
            ),
            M.Lane(
                1, "1", 19087, 262144, Path("lane1.json"),
                process=_AliveProcess(),
                live_request_bytes=15335,
                live_sequence=12,
                live_prompt_signature=(("shared", 14989), ("lane1", 346)),
            ),
            M.Lane(
                2, "2", 19088, 262144, Path("lane2.json"),
                process=_AliveProcess(),
                live_request_bytes=15187,
                live_sequence=10,
                live_prompt_signature=(("shared", 14989), ("lane2", 198)),
            ),
        ]
        pool = M.LanePool(
            lanes,
            scheduler_policy="balanced-additive-new-prefill-retained-state-proxy-v1",
        )
        pool.affinity.update({
            "old-0a": 0,
            "old-0b": 0,
            "old-1a": 1,
            "old-1b": 1,
            "old-2a": 2,
            "old-2b": 2,
        })
        incoming = (("shared", 14989), ("new", 123))
        selected = []
        for i in range(6):
            key = f"new-{i}"
            lane = pool.acquire(
                affinity_key=key,
                request_bytes=15112,
                prompt_signature=incoming,
            )
            selected.append(lane.index)
            pool.release(
                lane,
                affinity_key=key,
                request_bytes=15112,
                prompt_signature=incoming,
            )

        self.assertEqual({index: selected.count(index) for index in range(3)}, {0: 2, 1: 2, 2: 2})
        affinity_counts = {
            index: sum(1 for lane_index in pool.affinity.values() if lane_index == index)
            for index in range(3)
        }
        self.assertEqual(affinity_counts, {0: 4, 1: 4, 2: 4})

    def test_benchmark_policy_does_not_override_existing_session_affinity(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes, scheduler_policy="cache-aware-idle-v1")
        pool.affinity["returning"] = 1
        lanes[0].live_prompt_signature = (("perfect-match", 100),)
        decision = {}
        got = pool.acquire(
            affinity_key="returning",
            prompt_signature=(("perfect-match", 100),),
            decision_out=decision,
        )
        self.assertEqual(got.index, 1)
        self.assertEqual(decision["selected_reason"], "session_affinity")
        self.assertEqual(decision["policy"], "cache-aware-idle-v1")
        pool.release(got, affinity_key="returning")

    def test_unknown_scheduler_policy_is_rejected(self):
        lane = M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess())
        with self.assertRaisesRegex(ValueError, "unknown scheduler policy"):
            M.LanePool([lane], scheduler_policy="mystery")

    def test_unidentified_live_state_is_not_treated_as_empty(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(),
                   live_affinity_key=None, live_request_bytes=5000, live_sequence=1),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(),
                   live_affinity_key="small", live_request_bytes=1000, live_sequence=2),
        ]
        pool = M.LanePool(lanes)
        got = pool.acquire(affinity_key="new-chat", request_bytes=500)
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-chat", request_bytes=500)

    def test_new_session_prefers_empty_live_lane(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(),
                   live_affinity_key="long", live_request_bytes=200000),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(),
                   live_affinity_key="medium", live_request_bytes=60000),
        ]
        pool = M.LanePool(lanes)
        got = pool.acquire(affinity_key="new-chat", request_bytes=500)
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-chat", request_bytes=500)

    def test_new_session_prefers_smallest_live_state_without_lane_hardcoding(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(),
                   live_affinity_key="long", live_request_bytes=200000),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(),
                   live_affinity_key="medium", live_request_bytes=60000),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(),
                   live_affinity_key="short", live_request_bytes=700),
        ]
        pool = M.LanePool(lanes)
        got = pool.acquire(affinity_key="new-chat", request_bytes=500)
        self.assertEqual(got.index, 2)
        pool.release(got, affinity_key="new-chat", request_bytes=500)

    def test_equal_size_prefers_least_recent_live_state(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(),
                   live_affinity_key="a", live_request_bytes=1000, live_sequence=30),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(),
                   live_affinity_key="b", live_request_bytes=1000, live_sequence=10),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(),
                   live_affinity_key="c", live_request_bytes=1000, live_sequence=20),
        ]
        pool = M.LanePool(lanes)
        got = pool.acquire(affinity_key="new-chat", request_bytes=500)
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-chat", request_bytes=500)

    def test_equal_live_states_use_rotating_candidate_order(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess(),
                   live_affinity_key="a", live_request_bytes=1000),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess(),
                   live_affinity_key="b", live_request_bytes=1000),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess(),
                   live_affinity_key="c", live_request_bytes=1000),
        ]
        pool = M.LanePool(lanes)
        pool.cursor = 1
        got = pool.acquire(affinity_key="new-chat", request_bytes=500)
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-chat", request_bytes=500)

    def test_other_session_does_not_erase_existing_lane_affinity(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
            M.Lane(2, "2", 19088, 262144, Path("lane2.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        first = pool.acquire(affinity_key="long-chat", request_bytes=200000)
        self.assertEqual(first.index, 0)
        pool.release(first, affinity_key="long-chat", request_bytes=200000)
        for key, size in (("chat-b", 60000), ("chat-c", 700)):
            lane = pool.acquire(affinity_key=key, request_bytes=size)
            pool.release(lane, affinity_key=key, request_bytes=size)
        lane = pool.acquire(affinity_key="short-chat", request_bytes=500)
        self.assertEqual(lane.index, 2)
        pool.release(lane, affinity_key="short-chat", request_bytes=500)
        self.assertEqual(pool.affinity["long-chat"], 0)
        again = pool.acquire(affinity_key="long-chat", request_bytes=201000)
        self.assertEqual(again.index, 0)
        pool.release(again, affinity_key="long-chat", request_bytes=201000)

    def test_affinity_lru_is_bounded(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes, max_affinity_entries=3)
        for key in ("a", "b", "c", "d"):
            lane = pool.acquire(affinity_key=key)
            pool.release(lane)
        self.assertNotIn("a", pool.affinity)
        self.assertEqual(set(pool.affinity), {"b", "c", "d"})

    def test_vision_request_rebinds_affinity_to_a_vision_lane(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), vision=False, process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), vision=True, process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.affinity["chat-a"] = 0
        got = pool.acquire(requires_vision=True, affinity_key="chat-a")
        self.assertEqual(got.index, 1)
        self.assertEqual(pool.affinity["chat-a"], 1)
        pool.release(got)

    def test_metadata_prefers_healthy_vision_lane(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), vision=False, process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), vision=True, process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        old = M.lane_engine_alive
        M.lane_engine_alive = lambda lane: lane.index == 1
        try:
            self.assertEqual(pool.metadata_lane(lanes[0]).index, 1)
        finally:
            M.lane_engine_alive = old
        self.assertIn("/health", M.VISION_METADATA_PATHS)

    def test_status_distinguishes_wrapper_from_child_engine_health(self):
        lane = M.Lane(0, "0", 19086, 262144, Path("lane0.json"), vram_reserve_mib=1200,
                      process=_AliveProcess())
        pool = M.LanePool([lane])
        old = M.lane_engine_alive
        M.lane_engine_alive = lambda _: False
        try:
            status = pool.status()[0]
        finally:
            M.lane_engine_alive = old
        self.assertTrue(status["wrapper_alive"])
        self.assertFalse(status["alive"])
        self.assertEqual(status["vram_reserve_mib"], 1200)

    def test_acquire_excludes_wrapper_alive_child_dead_lane(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        old = M.lane_engine_alive
        M.lane_engine_alive = lambda lane: lane.index == 1
        try:
            got = pool.acquire(affinity_key="new-chat")
        finally:
            M.lane_engine_alive = old
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-chat")

    def test_new_session_does_not_erase_affinity_owned_by_live_wrapper_with_dead_child(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.affinity["returning"] = 0
        old_alive = M.lane_engine_alive
        M.lane_engine_alive = lambda lane: lane.index == 1
        try:
            got = pool.acquire(affinity_key="new-session")
        finally:
            M.lane_engine_alive = old_alive
        self.assertEqual(got.index, 1)
        pool.release(got, affinity_key="new-session")
        self.assertEqual(pool.affinity.get("returning"), 0)

    def test_returning_affinity_can_select_wrapper_ready_child_dead_lane_for_restart(self):
        lanes = [
            M.Lane(0, "0", 19086, 262144, Path("lane0.json"), process=_AliveProcess()),
            M.Lane(1, "1", 19087, 262144, Path("lane1.json"), process=_AliveProcess()),
        ]
        pool = M.LanePool(lanes)
        pool.affinity["returning"] = 0
        old_alive, old_ready = M.lane_engine_alive, M.lane_service_ready
        M.lane_engine_alive = lambda lane: lane.index == 1
        M.lane_service_ready = lambda lane: lane.index == 0
        decision = {}
        try:
            got = pool.acquire(affinity_key="returning", decision_out=decision)
        finally:
            M.lane_engine_alive, M.lane_service_ready = old_alive, old_ready
        self.assertEqual(got.index, 0)
        self.assertEqual(decision["selected_reason"], "session_affinity_restart")
        pool.release(got, affinity_key="returning")

    def test_native_arena_size_matches_arena_expert_source_contract(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "native_experts.txt").write_text(
                "# strata native experts v3 (n_expert 512, total 153600)\n"
                "0 42 2 0 100 10 20 30\n"
                "1 42 2 51200 200 40 50 60\n",
                encoding="utf-8",
            )
            got = M.native_arena_spec(p)
            self.assertEqual(got.n_expert, 512)
            self.assertEqual(got.max_blob, 200)
            self.assertEqual(got.expert_bytes, 51200 + 200 * 512)
            self.assertEqual(got.bytes, got.expert_bytes + got.max_blob)

    def test_512k_partition_for_three_lanes(self):
        got = M.lane_contexts(131072, 3, "262144,131072,131072", 524288)
        self.assertEqual(got, [262144, 131072, 131072])
        self.assertEqual(sum(got), 524288)

    def test_even_partition_when_only_budget_is_given(self):
        got = M.lane_contexts(131072, 3, None, 524288)
        self.assertEqual(sum(got), 524288)
        self.assertLessEqual(max(got) - min(got), 1)

    def test_reject_contexts_over_budget(self):
        with self.assertRaisesRegex(ValueError, "above --kv-budget"):
            M.lane_contexts(131072, 3, "262144,262144,131072", 524288)

    def test_reject_lane_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "3 GPU lanes"):
            M.lane_contexts(131072, 3, "262144,131072", 524288)

    def test_cpu_partition_keeps_smt_siblings_together(self):
        groups = [(i, i + 16) for i in range(16)]
        got = M.partition_cpu_sets(groups, 3)
        self.assertEqual([len(x) for x in got], [12, 10, 10])
        self.assertEqual(set().union(*(set(x) for x in got)), set(range(32)))
        self.assertTrue(set(got[0]).isdisjoint(got[1]))
        self.assertTrue(set(got[0]).isdisjoint(got[2]))
        self.assertTrue(set(got[1]).isdisjoint(got[2]))
        for i in range(16):
            owners = [n for n, cpus in enumerate(got) if i in cpus or i + 16 in cpus]
            self.assertEqual(len(owners), 1)
            self.assertIn(i, got[owners[0]])
            self.assertIn(i + 16, got[owners[0]])

    def test_cpu_partition_needs_host_and_worker_core_per_lane(self):
        with self.assertRaisesRegex(ValueError, "two physical cores per lane"):
            M.partition_cpu_sets([(0,), (1,), (2,), (3,), (4,)], 3)

    def test_exact_cpu_partition_biases_middle_lane(self):
        groups = [(i, i + 16) for i in range(16)]
        got = M.partition_cpu_sets_exact(groups, [5, 6, 5])
        self.assertEqual([len(x) for x in got], [10, 12, 10])
        self.assertEqual(set().union(*(set(x) for x in got)), set(range(32)))
        for i in range(16):
            owners = [n for n, cpus in enumerate(got) if i in cpus or i + 16 in cpus]
            self.assertEqual(len(owners), 1)
            self.assertIn(i, got[owners[0]])
            self.assertIn(i + 16, got[owners[0]])

    def test_exact_cpu_partition_requires_full_budget(self):
        with self.assertRaisesRegex(ValueError, "totals 15 physical cores"):
            M.partition_cpu_sets_exact([(i,) for i in range(16)], [5, 5, 5])

    def test_parse_lane_pcie_fractions(self):
        self.assertEqual(M.parse_float_list("0.55,0.30,0.75", what="--lane-pcie-fracs"), [0.55, 0.30, 0.75])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            M.parse_float_list("0.55,1.2,0.55", what="--lane-pcie-fracs")


if __name__ == "__main__":
    unittest.main()

    def test_lane_parking_status_extracts_latest_engine_truth(self):
        lane = types.SimpleNamespace(index=0, port=19000, process=types.SimpleNamespace(poll=lambda: None))
        payload = json.dumps({"requests": [{"parked_conversations": 3, "parked_bytes": 1234, "park_evictions": 2}]}).encode()
        class R:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self): return payload
        with mock.patch.object(M, "lane_engine_alive", return_value=True), mock.patch.object(M.urllib.request, "urlopen", return_value=R()):
            self.assertEqual(M.lane_parking_status(lane), {"parked_conversations": 3, "parked_bytes": 1234, "park_evictions": 2})
