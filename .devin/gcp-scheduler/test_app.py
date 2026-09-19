import datetime
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import app


_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_KEY_PEM = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")

ENV = {
    "GITHUB_APP_ID": "123456",
    "GITHUB_APP_INSTALLATION_ID": "789012",
    "GITHUB_APP_PRIVATE_KEY": _PRIVATE_KEY_PEM,
}

EXPECTED_URL_ALERTS = (
    "https://api.github.com/repos/klagavyn-gif/tradV2"
    "/actions/workflows/main.yml/dispatches"
)
EXPECTED_URL_SUMMARY = (
    "https://api.github.com/repos/klagavyn-gif/tradV2"
    "/actions/workflows/daily-summary.yml/dispatches"
)
EXPECTED_URL_HEARTBEAT = (
    "https://api.github.com/repos/klagavyn-gif/tradV2"
    "/actions/workflows/heartbeat-check.yml/dispatches"
)
EXPECTED_URL_ENTRY_EDGE = (
    "https://api.github.com/repos/klagavyn-gif/tradV2"
    "/actions/workflows/entry-edge-report.yml/dispatches"
)
EXPECTED_INSTALLATION_URL = (
    "https://api.github.com/app/installations/789012/access_tokens"
)
EXPECTED_RUNS_URL = (
    "https://api.github.com/repos/klagavyn-gif/tradV2"
    "/actions/workflows/main.yml/runs?event=workflow_dispatch&branch=main&per_page=5"
)
EXPECTED_INPUTS = {
    "symbols": "BTC-USD,DOGE-USD,ETH-USD,ADA-USD,XRP-USD,BNB-USD,SOL-USD,TRX-USD,NEAR-USD,LINK-USD,PAXG-USD",
    "period": "15m",
    "retry_attempt": "0",
    "retry_source_run_id": "",
    "retry_reason": "google_cloud_scheduler",
}

SECRETS = ["123456", "789012", "secret-installation-token", _PRIVATE_KEY_PEM]


class FakeResponse:
    def __init__(self, status, payload=b""):
        self.status = status
        self._payload = payload
        self.closed = False
        self.read_size = None

    def read(self, size=-1):
        self.read_size = size
        return self._payload

    def close(self):
        self.closed = True


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(b"upstream body"))


def _recent_runs_response(run_id=123456, created_at=None):
    if created_at is None:
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "workflow_runs": [
            {
                "id": run_id,
                "event": "workflow_dispatch",
                "head_branch": "main",
                "created_at": created_at,
            }
        ]
    }
    return FakeResponse(200, json.dumps(payload).encode())


def _empty_runs_response():
    return FakeResponse(200, json.dumps({"workflow_runs": []}).encode())


def _assert_no_secret_leak(testcase, exc):
    text = str(exc) + repr(exc)
    for secret in SECRETS:
        testcase.assertNotIn(secret, text)


class AppJwtTests(unittest.TestCase):
    def test_jwt_claims_exact(self):
        token = app.create_app_jwt(environ=ENV, now=1_700_000_000)
        decoded = jwt.decode(token, options={"verify_signature": False})
        self.assertEqual(decoded["iat"], 1_700_000_000 - 60)
        self.assertEqual(decoded["exp"], 1_700_000_000 + 540)
        self.assertEqual(decoded["iss"], "123456")

    def test_invalid_private_key_sanitized(self):
        env = dict(ENV)
        env["GITHUB_APP_PRIVATE_KEY"] = "not-a-real-key-value"
        with self.assertRaises(app.DispatchError) as ctx:
            app.create_app_jwt(environ=env)
        self.assertFalse(ctx.exception.retryable)
        self.assertNotIn("not-a-real-key-value", str(ctx.exception))


class InstallationTokenTests(unittest.TestCase):
    def test_installation_request_exact(self):
        url, headers, body = app.build_installation_token_request(
            environ=ENV, now=1_700_000_000
        )
        self.assertEqual(url, EXPECTED_INSTALLATION_URL)
        self.assertEqual(headers["Accept"], "application/vnd.github+json")
        self.assertTrue(headers["Authorization"].startswith("Bearer "))
        decoded_jwt = jwt.decode(
            headers["Authorization"][7:], options={"verify_signature": False}
        )
        self.assertEqual(decoded_jwt["iss"], "123456")
        self.assertEqual(headers["X-GitHub-Api-Version"], "2026-03-10")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["User-Agent"], "tradv2-gcp-scheduler/1.0")
        decoded = json.loads(body.decode("utf-8"))
        self.assertEqual(
            decoded,
            {"repositories": ["tradV2"], "permissions": {"actions": "write"}},
        )

    def test_missing_or_blank_app_env_rejected(self):
        for env_name in (
            "GITHUB_APP_ID",
            "GITHUB_APP_INSTALLATION_ID",
            "GITHUB_APP_PRIVATE_KEY",
        ):
            for supplied in (None, "   "):
                env = dict(ENV)
                if supplied is None:
                    env.pop(env_name)
                else:
                    env[env_name] = supplied
                with self.assertRaises(app.DispatchError) as ctx:
                    app.build_installation_token_request(environ=env)
                self.assertFalse(ctx.exception.retryable, env_name)
                if supplied:
                    self.assertNotIn(supplied, str(ctx.exception))
                self.assertIn(env_name, str(ctx.exception))

    def test_nondigit_installation_id_rejected(self):
        env = dict(ENV)
        env["GITHUB_APP_INSTALLATION_ID"] = "78x012"
        with self.assertRaises(app.DispatchError) as ctx:
            app.build_installation_token_request(environ=env)
        self.assertFalse(ctx.exception.retryable)
        self.assertNotIn("78x012", str(ctx.exception))

    def test_token_response_parsed_and_closed(self):
        response = FakeResponse(201, json.dumps({"token": "secret-installation-token"}).encode())

        def opener(request, timeout=None):
            self.assertEqual(timeout, 15)
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, EXPECTED_INSTALLATION_URL)
            return response

        token = app.request_installation_token(environ=ENV, opener=opener, now=1_700_000_000)
        self.assertEqual(token, "secret-installation-token")
        self.assertTrue(response.closed)
        self.assertLessEqual(response.read_size, 65536)

    def test_malformed_token_response_rejected(self):
        for payload in (b"not json", json.dumps({"nope": 1}).encode(),
                        json.dumps({"token": "   "}).encode()):
            response = FakeResponse(201, payload)
            with self.assertRaises(app.DispatchError) as ctx:
                app.request_installation_token(
                    environ=ENV,
                    opener=lambda request, timeout=None: response,
                    now=1_700_000_000,
                )
            self.assertTrue(response.closed)
            self.assertNotIn(payload.decode("utf-8", "replace"), str(ctx.exception))

    def test_installation_http_error_retryability(self):
        for code, retryable in ((403, False), (500, True)):
            def opener(request, timeout=None):
                raise _http_error(request.full_url, code)

            with self.assertRaises(app.DispatchError) as ctx:
                app.request_installation_token(
                    environ=ENV, opener=opener, now=1_700_000_000
                )
            self.assertEqual(ctx.exception.retryable, retryable, code)
            self.assertNotIn("upstream body", str(ctx.exception))
            _assert_no_secret_leak(self, ctx.exception)


class DispatchRequestTests(unittest.TestCase):
    def test_alerts_request_exact(self):
        url, headers, body = app.build_dispatch_request(
            "alerts", "secret-installation-token", environ=ENV
        )
        self.assertEqual(url, EXPECTED_URL_ALERTS)
        self.assertEqual(headers["Accept"], "application/vnd.github+json")
        self.assertEqual(headers["Authorization"], "Bearer secret-installation-token")
        self.assertEqual(headers["X-GitHub-Api-Version"], "2026-03-10")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertTrue(headers["User-Agent"].strip())
        decoded = json.loads(body.decode("utf-8"))
        self.assertEqual(decoded, {"ref": "main", "inputs": dict(EXPECTED_INPUTS)})
        self.assertNotIn("force", decoded["inputs"])

    def test_daily_summary_includes_force(self):
        url, headers, body = app.build_dispatch_request(
            "daily-summary", "secret-installation-token", environ=ENV
        )
        self.assertEqual(url, EXPECTED_URL_SUMMARY)
        decoded = json.loads(body.decode("utf-8"))
        expected = dict(EXPECTED_INPUTS)
        expected["force"] = "true"
        self.assertEqual(decoded, {"ref": "main", "inputs": expected})

    def test_heartbeat_request_has_no_inputs(self):
        url, headers, body = app.build_dispatch_request(
            "heartbeat", "secret-installation-token", environ=ENV
        )
        self.assertEqual(url, EXPECTED_URL_HEARTBEAT)
        self.assertEqual(json.loads(body.decode("utf-8")), {"ref": "main"})

    def test_entry_edge_request_inputs(self):
        url, headers, body = app.build_dispatch_request(
            "entry-edge-report", "secret-installation-token", environ=ENV
        )
        self.assertEqual(url, EXPECTED_URL_ENTRY_EDGE)
        decoded = json.loads(body.decode("utf-8"))
        self.assertEqual(decoded["ref"], "main")
        self.assertEqual(
            decoded["inputs"],
            {
                "days": "45",
                "since": "",
                "cost_bps": "30",
                "target_settled": "100",
                "notify": "true",
            },
        )

    def test_entry_edge_request_env_overrides(self):
        env = dict(ENV)
        env["ENTRY_EDGE_DAYS"] = "30"
        env["ENTRY_EDGE_SINCE"] = "2026-09-17"
        env["ENTRY_EDGE_COST_BPS"] = "25"
        env["ENTRY_EDGE_TARGET_SETTLED"] = "80"
        env["ENTRY_EDGE_NOTIFY"] = "false"
        _, _, body = app.build_dispatch_request(
            "entry-edge-report", "secret-installation-token", environ=env
        )
        decoded = json.loads(body.decode("utf-8"))
        self.assertEqual(
            decoded["inputs"],
            {
                "days": "30",
                "since": "2026-09-17",
                "cost_bps": "25",
                "target_settled": "80",
                "notify": "false",
            },
        )

    def test_unsupported_workflow_raises(self):
        with self.assertRaises(app.DispatchError):
            app.build_dispatch_request("nope", "secret-installation-token", environ=ENV)

    def test_blank_destination_env_raises_nonretryable(self):
        for env_name in (
            "GITHUB_OWNER",
            "GITHUB_REPO",
            "GITHUB_REF",
            "TRADV2_SYMBOLS",
            "TRADV2_PERIOD",
        ):
            for blank in ("   ", "\t\n"):
                env = dict(ENV)
                env[env_name] = blank
                with self.assertRaises(app.DispatchError) as ctx:
                    app.build_dispatch_request(
                        "alerts", "secret-installation-token", environ=env
                    )
                self.assertFalse(ctx.exception.retryable, env_name)
                self.assertNotIn(blank, str(ctx.exception))
                self.assertIn(env_name, str(ctx.exception))

    def test_blank_installation_token_rejected(self):
        with self.assertRaises(app.DispatchError) as ctx:
            app.build_dispatch_request("alerts", "   ", environ=ENV)
        self.assertFalse(ctx.exception.retryable)


class DispatchWorkflowTests(unittest.TestCase):
    def test_full_dispatch_sequence(self):
        calls = []
        responses = [
            FakeResponse(201, json.dumps({"token": "secret-installation-token"}).encode()),
            FakeResponse(204),
        ]

        def opener(request, timeout=None):
            calls.append(request.full_url)
            return responses[len(calls) - 1]

        result = app.dispatch_workflow(
            "alerts", environ=ENV, opener=opener, now=1_700_000_000
        )
        self.assertEqual(result, {"workflow": "alerts", "status": "accepted"})
        self.assertEqual(
            calls, [EXPECTED_INSTALLATION_URL, EXPECTED_URL_ALERTS]
        )
        self.assertTrue(all(r.closed for r in responses))

    def test_workflow_other_2xx_accepted(self):
        responses = [
            FakeResponse(201, json.dumps({"token": "secret-installation-token"}).encode()),
            FakeResponse(202),
        ]
        calls = []

        def opener(request, timeout=None):
            calls.append(request.full_url)
            return responses[len(calls) - 1]

        result = app.dispatch_workflow(
            "alerts", environ=ENV, opener=opener, now=1_700_000_000
        )
        self.assertEqual(result, {"workflow": "alerts", "status": "accepted"})
        self.assertEqual(calls, [EXPECTED_INSTALLATION_URL, EXPECTED_URL_ALERTS])
        self.assertTrue(all(r.closed for r in responses))

    def test_workflow_5xx_without_recorded_run_is_retryable(self):
        responses = [
            FakeResponse(201, json.dumps({"token": "tok"}).encode()),
            FakeResponse(500),
            _empty_runs_response(),
        ]
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            return responses[len(calls) - 1]

        with self.assertRaises(app.DispatchError) as ctx:
            app.dispatch_workflow("alerts", environ=ENV, opener=opener, now=1_700_000_000)
        self.assertTrue(ctx.exception.retryable)
        self.assertTrue(all(r.closed for r in responses))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2].full_url, EXPECTED_RUNS_URL)
        self.assertEqual(calls[2].get_method(), "GET")

    def test_network_error_confirmed_by_recent_run_is_accepted(self):
        responses = [
            FakeResponse(201, json.dumps({"token": "tok"}).encode()),
            _recent_runs_response(run_id=987654),
        ]
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            if len(calls) == 2:
                raise urllib.error.URLError("connection lost after request")
            return responses[0] if len(calls) == 1 else responses[1]

        result = app.dispatch_workflow(
            "alerts", environ=ENV, opener=opener, now=1_700_000_000
        )
        self.assertEqual(
            result,
            {"workflow": "alerts", "status": "accepted", "run_id": "987654"},
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2].full_url, EXPECTED_RUNS_URL)

    def test_network_error_without_recent_run_is_retryable(self):
        responses = [
            FakeResponse(201, json.dumps({"token": "tok"}).encode()),
            _empty_runs_response(),
        ]
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            if len(calls) == 2:
                raise urllib.error.URLError("connection lost")
            return responses[0] if len(calls) == 1 else responses[1]

        with self.assertRaises(app.DispatchError) as ctx:
            app.dispatch_workflow("alerts", environ=ENV, opener=opener, now=1_700_000_000)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(len(calls), 3)

    def test_network_error_with_failed_verification_is_not_retried(self):
        calls = []

        def opener(request, timeout=None):
            calls.append(request)
            if len(calls) == 1:
                return FakeResponse(201, json.dumps({"token": "tok"}).encode())
            if len(calls) == 2:
                raise urllib.error.URLError("connection lost")
            return FakeResponse(500)

        with self.assertRaises(app.DispatchError) as ctx:
            app.dispatch_workflow("alerts", environ=ENV, opener=opener, now=1_700_000_000)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(calls), 3)


class HandlerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), app.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(self, method, path, payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            "http://127.0.0.1:{}{}".format(self.port, path),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=10)
            return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_http_endpoints(self):
        status, body = self._request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

        status, body = self._request("POST", "/dispatch", {"workflow": "bogus"})
        self.assertEqual(status, 400)

        status, body = self._request("GET", "/nope")
        self.assertEqual(status, 404)

        original = app.dispatch_workflow
        calls = []

        def fake_dispatch(workflow, **kwargs):
            calls.append(workflow)
            return {"workflow": workflow, "status": "accepted"}

        app.dispatch_workflow = fake_dispatch
        try:
            status, body = self._request("POST", "/dispatch", {"workflow": "alerts"})
        finally:
            app.dispatch_workflow = original
        self.assertEqual(status, 202)
        self.assertEqual(body, {"workflow": "alerts", "status": "accepted"})
        self.assertEqual(calls, ["alerts"])

        def raise_nonretryable(workflow, **kwargs):
            raise app.DispatchError("denied", retryable=False)

        app.dispatch_workflow = raise_nonretryable
        try:
            status, body = self._request("POST", "/dispatch", {"workflow": "alerts"})
        finally:
            app.dispatch_workflow = original
        self.assertEqual(status, 502)

        def raise_retryable(workflow, **kwargs):
            raise app.DispatchError("upstream busy", retryable=True)

        app.dispatch_workflow = raise_retryable
        try:
            status, body = self._request("POST", "/dispatch", {"workflow": "alerts"})
        finally:
            app.dispatch_workflow = original
        self.assertEqual(status, 503)


if __name__ == "__main__":
    unittest.main()
