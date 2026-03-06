from __future__ import annotations

import json
import threading
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from kis_paper import KISPaperClient, RequestThrottle


class _FakeClock:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        with self._lock:
            return self._value

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self._value += seconds


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)
        self.headers = {}

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, response_payload: dict | None = None) -> None:
        self.response_payload = response_payload or {}
        self.post_calls: list[tuple[str, dict | None]] = []
        self.get_calls: list[tuple[str, dict | None]] = []

    def post(self, url: str, headers=None, data=None, timeout=None):  # noqa: ANN001
        self.post_calls.append((url, headers))
        return _FakeResponse(self.response_payload, status_code=200)

    def get(self, url: str, headers=None, params=None, timeout=None):  # noqa: ANN001
        self.get_calls.append((url, params))
        return _FakeResponse({}, status_code=200)


def _write_kis_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "my_app: test_app_key",
                "my_sec: test_app_secret",
                "my_acct: 12345678",
                "my_prod: 01",
                "prod: https://openapivts.koreainvestment.com:29443",
                "my_agent: codex-test",
            ]
        ),
        encoding="utf-8",
    )


class KISPaperClientTest(unittest.TestCase):
    def test_request_throttle_is_thread_safe(self) -> None:
        clock = _FakeClock()
        throttle = RequestThrottle(0.1, monotonic_fn=clock.monotonic, sleep_fn=clock.sleep)
        threads = [threading.Thread(target=throttle.wait) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(clock.sleeps), 3)
        self.assertGreaterEqual(sum(clock.sleeps), 0.3)

    def test_get_access_token_uses_valid_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "kis.yaml"
            _write_kis_config(config_path)
            data_dir = root / "data"
            data_dir.mkdir()
            cache_path = data_dir / "kis_token_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "config_id": "https://openapivts.koreainvestment.com:29443|12345678|01|pp_key",
                        "access_token": "cached-token",
                        "expires_at": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                ),
                encoding="utf-8",
            )
            session = _FakeSession()
            client = KISPaperClient(config_path=config_path, data_dir=data_dir, session=session)
            # config_id uses last 8 chars of app key: "app_key" from "test_app_key"
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["config_id"] = client.config.config_id
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(client.get_access_token(), "cached-token")
            self.assertEqual(session.post_calls, [])

    def test_get_access_token_refreshes_and_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "kis.yaml"
            _write_kis_config(config_path)
            session = _FakeSession(
                {
                    "access_token": "fresh-token",
                    "access_token_token_expired": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            client = KISPaperClient(config_path=config_path, data_dir=root / "data", session=session)
            token = client.get_access_token(force_refresh=True)
            self.assertEqual(token, "fresh-token")
            self.assertEqual(len(session.post_calls), 1)
            payload = json.loads(client.token_cache_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["access_token"], "fresh-token")
            self.assertEqual(payload["config_id"], client.config.config_id)


if __name__ == "__main__":
    unittest.main()
