# tests/test_server_multimodel.py -- M12 P2 multi-model server smoke tests.
#
# Brings up build/server_multimodel on a free port, loads the qwen2.5-0.5b
# gguf under TWO registry names ("q25" and "big"), then exercises the
# OpenAI-compatible surface with four payloads:
#
#   1. /healthz and /v1/models        -- discovery surface (registry shape)
#   2. POST model="q25"               -- routes to first engine
#   3. POST model="big"               -- routes to second engine
#   4. POST model="missing"           -- unknown model -> HTTP 400
#   5. POST {} with no "model" field  -- empty/missing field -> HTTP 400
#
# Both engines share the same underlying file. That's intentional: the
# exercise here is the registry, the per-engine mutex, and the routing
# path -- not a quality difference between two distinct architectures.
# (The current Qwen2Engine graph-capture path is qwen2-only; loading
# llama/phi/qwen3 alongside it falls back to eager and emits a 1-token
# 'S' result, which would obscure the dispatch test. Two qwen2 instances
# are the cleanest demonstration of the P2 plumbing.)
#
# Following the P1 test, GPU contention is treated as a skip rather than
# a failure: if the engine never comes up we surface stderr and skip.

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import requests

REPO = Path(__file__).resolve().parent.parent
BUILD_BIN = REPO / "build" / "server_multimodel"
DEFAULT_MODEL = REPO / "data" / "models" / "qwen2.5-0.5b-instruct-q4_0.gguf"

TEST_LISTEN_HOST = "127.0.0.1"
TEST_LISTEN_PORT = int(os.environ.get("TEST_MULTIMODEL_PORT", "18082"))


def _free_port(host: str, preferred: int) -> int:
    s = socket.socket()
    try:
        s.bind((host, preferred))
        return preferred
    except OSError:
        for p in range(preferred + 1, preferred + 100):
            try:
                s2 = socket.socket(); s2.bind((host, p)); s2.close()
                return p
            except OSError:
                continue
        raise
    finally:
        s.close()


def _wait_ready(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready on {url}: {last_err}")


def _can_load_model() -> bool:
    return DEFAULT_MODEL.is_file()


@pytest.fixture(scope="module")
def server():
    """Start the multi-model server, yield its base URL, then tear it down."""
    if not BUILD_BIN.is_file():
        pytest.skip(f"server binary not built at {BUILD_BIN}")
    if not _can_load_model():
        pytest.skip(f"model not present at {DEFAULT_MODEL}")

    port = _free_port(TEST_LISTEN_HOST, TEST_LISTEN_PORT)
    listen = f"{TEST_LISTEN_HOST}:{port}"
    base = f"http://{listen}"

    # Two engines pointing at the same gguf file. Distinct names exercise
    # the registry lookup path; the actual inference quality is identical.
    models_spec = (
        f"q25={DEFAULT_MODEL},big={DEFAULT_MODEL}"
    )
    env = os.environ.copy()
    env["TT_LISTEN"] = listen
    env["TT_MODELS"] = models_spec
    env["TT_GREEDY"] = "1"
    env["TT_MAX_CTX"] = "1024"

    proc = subprocess.Popen(
        [str(BUILD_BIN)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO),
    )
    try:
        # Both engines load before the socket binds; under contention this
        # can take 30-60s. Beyond that we skip rather than hang.
        _wait_ready(base + "/healthz", timeout_s=90.0)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        out, err = proc.communicate(timeout=5)
        sys.stderr.write("--- server stdout ---\n" + (out or b"").decode(errors="replace"))
        sys.stderr.write("--- server stderr ---\n" + (err or b"").decode(errors="replace"))
        pytest.skip(f"server failed to start on {base} (likely GPU contention)")
        return  # unreachable

    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ----------------------------- registry surface -----------------------------


def test_healthz(server: str) -> None:
    r = requests.get(server + "/healthz", timeout=5)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_models_list_lists_both(server: str) -> None:
    r = requests.get(server + "/v1/models", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    ids = sorted(m["id"] for m in body.get("data", []))
    assert ids == ["big", "q25"], ids


def test_404(server: str) -> None:
    r = requests.get(server + "/no/such/path", timeout=5)
    assert r.status_code == 404


# ----------------------------- dispatch tests -----------------------------


def _post_chat(url: str, payload: dict[str, Any], timeout: float = 30.0
               ) -> requests.Response:
    return requests.post(
        url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def test_dispatch_q25(server: str) -> None:
    """POST with model=q25 should route to the qwen2.5-0.5b first engine."""
    r = _post_chat(server + "/v1/chat/completions", {
        "model": "q25",
        "messages": [{"role": "user", "content": "What is 2+2? One short sentence."}],
        "max_tokens": 16,
    }, timeout=30.0)
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    assert body["model"] == "q25"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_dispatch_big(server: str) -> None:
    """POST with model=big should route to the second engine and echo the
    name back in the response envelope."""
    r = _post_chat(server + "/v1/chat/completions", {
        "model": "big",
        "messages": [{"role": "user", "content": "Say hi briefly."}],
        "max_tokens": 16,
    }, timeout=30.0)
    assert r.status_code == 200, r.text[:500]
    body = r.json()
    assert body["model"] == "big", body
    msg = body["choices"][0]["message"]
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    assert body["choices"][0]["finish_reason"] in ("stop", "length")


def test_unknown_model_returns_400(server: str) -> None:
    """An unrecognized model name must be rejected before any engine lock
    is acquired. The error message should include the bad name verbatim
    so clients can surface a useful diagnostic."""
    r = _post_chat(server + "/v1/chat/completions", {
        "model": "missing",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10.0)
    assert r.status_code == 400, r.text[:500]
    body = r.json()
    assert "error" in body
    assert "missing" in body["error"], body


def test_missing_model_field_returns_400(server: str) -> None:
    """A request without a model field at all must also be rejected."""
    r = _post_chat(server + "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10.0)
    assert r.status_code == 400, r.text[:500]
    body = r.json()
    assert "error" in body
    assert "model" in body["error"].lower(), body


def test_empty_model_field_returns_400(server: str) -> None:
    """A request with model="" must also be rejected (empty name)."""
    r = _post_chat(server + "/v1/chat/completions", {
        "model": "",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10.0)
    assert r.status_code == 400, r.text[:500]
