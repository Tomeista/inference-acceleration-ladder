from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port: int):
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(base_url + "/health", timeout=1.0).status_code == 200:
                return base_url, server, thread
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError("mock llama-server did not become ready")


@pytest.fixture(scope="session")
def mock_llama() -> str:
    """A mock llama-server on a free port, torn down at the end of the session."""
    from mock_llama_server import app

    base_url, server, thread = _serve(app, _free_port())
    yield base_url
    server.should_exit = True
    thread.join(timeout=10)
