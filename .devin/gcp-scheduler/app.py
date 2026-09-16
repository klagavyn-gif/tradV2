import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, urlencode

import jwt

WORKFLOWS = {
    "alerts": "main.yml",
    "daily-summary": "daily-summary.yml",
}

_DEFAULT_SYMBOLS = "BTC-USD,DOGE-USD,ETH-USD,ADA-USD,XRP-USD,BNB-USD,SOL-USD,TRX-USD,NEAR-USD,LINK-USD,PAXG-USD"
_MAX_BODY_BYTES = 4096
_MAX_RESPONSE_BYTES = 65536
_MAX_RUNS_RESPONSE_BYTES = 1048576
_REQUEST_TIMEOUT = 15
_DISPATCH_CONFIRMATION_SKEW_SECONDS = 10
_API_VERSION = "2026-03-10"
_ACCEPT = "application/vnd.github+json"
_USER_AGENT = "tradv2-gcp-scheduler/1.0"
_NONRETRYABLE_HTTP_STATUSES = {400, 401, 403, 404, 422}


class DispatchError(RuntimeError):
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = bool(retryable)


def _log_event(event, **fields):
    record = {"event": str(event)}
    for key, value in fields.items():
        if value is not None:
            record[str(key)] = value
    sys.stderr.write(json.dumps(record, sort_keys=True) + "\n")
    sys.stderr.flush()


def _env(environ):
    return os.environ if environ is None else environ


def _required_env(env, name):
    value = str(env.get(name) or "").strip()
    if not value:
        raise DispatchError("{} is not configured".format(name), retryable=False)
    return value


def _resolve_destination(env):
    owner = str(env.get("GITHUB_OWNER") or "klagavyn-gif").strip()
    repo = str(env.get("GITHUB_REPO") or "tradV2").strip()
    ref = str(env.get("GITHUB_REF") or "main").strip()
    symbols = str(env.get("TRADV2_SYMBOLS") or _DEFAULT_SYMBOLS).strip()
    period = str(env.get("TRADV2_PERIOD") or "15m").strip()
    for env_name, value in (
        ("GITHUB_OWNER", owner),
        ("GITHUB_REPO", repo),
        ("GITHUB_REF", ref),
        ("TRADV2_SYMBOLS", symbols),
        ("TRADV2_PERIOD", period),
    ):
        if not value:
            raise DispatchError("{} is not configured".format(env_name), retryable=False)
    return owner, repo, ref, symbols, period


def _app_installation_id(env):
    installation_id = _required_env(env, "GITHUB_APP_INSTALLATION_ID")
    if not installation_id.isdigit():
        raise DispatchError("GITHUB_APP_INSTALLATION_ID is not configured", retryable=False)
    return installation_id


def _api_headers(token):
    return {
        "Accept": _ACCEPT,
        "Authorization": "Bearer " + token,
        "X-GitHub-Api-Version": _API_VERSION,
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }


def _http_error_status(exc):
    return getattr(exc, "code", None)


def _raise_sanitized_http(exc, context):
    status = _http_error_status(exc)
    raise DispatchError(
        "{} failed with status {}".format(context, status),
        retryable=status not in _NONRETRYABLE_HTTP_STATUSES,
    )


def _read_status_and_close(response):
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "code", None)
        return status
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def create_app_jwt(*, environ=None, now=None):
    """Return an RS256 GitHub App JWT, or raise DispatchError."""
    env = _env(environ)
    app_id = _required_env(env, "GITHUB_APP_ID")
    private_key = _required_env(env, "GITHUB_APP_PRIVATE_KEY")
    issued = time.time() if now is None else now
    payload = {
        "iat": int(issued) - 60,
        "exp": int(issued) + 540,
        "iss": app_id,
    }
    try:
        token = jwt.encode(payload, private_key, algorithm="RS256")
    except Exception:
        raise DispatchError("GitHub App private key is invalid", retryable=False)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def build_installation_token_request(*, environ=None, now=None):
    """Return (url, headers, body_bytes), or raise DispatchError."""
    env = _env(environ)
    installation_id = _app_installation_id(env)
    _, repo, _, _, _ = _resolve_destination(env)
    app_jwt = create_app_jwt(environ=env, now=now)
    url = "https://api.github.com/app/installations/{}/access_tokens".format(
        quote(installation_id, safe="")
    )
    headers = _api_headers(app_jwt)
    body = {
        "repositories": [repo],
        "permissions": {"actions": "write"},
    }
    return url, headers, json.dumps(body).encode("utf-8")


def request_installation_token(*, environ=None, opener=None, now=None):
    """Return a nonblank installation token, or raise DispatchError."""
    url, headers, body = build_installation_token_request(environ=environ, now=now)
    if opener is None:
        opener = urllib.request.urlopen
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        response = opener(request, timeout=_REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        _raise_sanitized_http(exc, "installation token request")
    except urllib.error.URLError:
        raise DispatchError("installation token request network error")
    except Exception:
        raise DispatchError("installation token request error")
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "code", None)
        if status != 201:
            raise DispatchError(
                "installation token request returned status {}".format(status),
                retryable=status not in _NONRETRYABLE_HTTP_STATUSES,
            )
        raw = response.read(_MAX_RESPONSE_BYTES)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not isinstance(payload, dict):
        raise DispatchError("installation token response was not a JSON object")
    token = str(payload.get("token") or "").strip()
    if not token:
        raise DispatchError("installation token is missing from response")
    return token


def build_dispatch_request(workflow_key, installation_token, *, environ=None):
    """Return (url, headers, body_bytes), or raise DispatchError."""
    env = _env(environ)
    key = str(workflow_key or "").strip()
    filename = WORKFLOWS.get(key)
    if filename is None:
        raise DispatchError("unsupported workflow", retryable=False)
    token = str(installation_token or "").strip()
    if not token:
        raise DispatchError("GitHub installation token is unavailable", retryable=False)
    owner, repo, ref, symbols, period = _resolve_destination(env)
    url = "https://api.github.com/repos/{}/{}/actions/workflows/{}/dispatches".format(
        quote(owner, safe=""),
        quote(repo, safe=""),
        quote(filename, safe=""),
    )
    headers = _api_headers(token)
    inputs = {
        "symbols": symbols,
        "period": period,
        "retry_attempt": "0",
        "retry_source_run_id": "",
        "retry_reason": "google_cloud_scheduler",
    }
    if key == "daily-summary":
        inputs["force"] = "true"
    body = {"ref": ref, "inputs": inputs}
    return url, headers, json.dumps(body).encode("utf-8")


def build_recent_runs_request(workflow_key, installation_token, *, environ=None):
    """Return (url, headers) for recent workflow_dispatch runs, or raise."""
    env = _env(environ)
    key = str(workflow_key or "").strip()
    filename = WORKFLOWS.get(key)
    if filename is None:
        raise DispatchError("unsupported workflow", retryable=False)
    token = str(installation_token or "").strip()
    if not token:
        raise DispatchError("GitHub installation token is unavailable", retryable=False)
    owner, repo, ref, _, _ = _resolve_destination(env)
    query = urlencode({
        "event": "workflow_dispatch",
        "branch": ref,
        "per_page": "5",
    })
    url = "https://api.github.com/repos/{}/{}/actions/workflows/{}/runs?{}".format(
        quote(owner, safe=""),
        quote(repo, safe=""),
        quote(filename, safe=""),
        query,
    )
    return url, _api_headers(token)


def _github_timestamp(value):
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def recent_dispatch_run_id(
    workflow_key,
    installation_token,
    *,
    environ=None,
    opener=None,
    since=None,
):
    """Return the ID of a matching recent workflow_dispatch run, or None."""
    env = _env(environ)
    if opener is None:
        opener = urllib.request.urlopen
    _, _, ref, _, _ = _resolve_destination(env)
    url, headers = build_recent_runs_request(
        workflow_key, installation_token, environ=env
    )
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        response = opener(request, timeout=_REQUEST_TIMEOUT)
    except Exception:
        raise DispatchError("github dispatch verification failed", retryable=False)
    try:
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "code", None)
        raw = response.read(_MAX_RUNS_RESPONSE_BYTES)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if status != 200:
        raise DispatchError("github dispatch verification failed", retryable=False)
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise DispatchError("github dispatch verification failed", retryable=False)
    threshold = (time.time() if since is None else since) - _DISPATCH_CONFIRMATION_SKEW_SECONDS
    for run in payload["workflow_runs"]:
        if not isinstance(run, dict):
            continue
        if str(run.get("event") or "") != "workflow_dispatch":
            continue
        if str(run.get("head_branch") or "") != ref:
            continue
        created = _github_timestamp(run.get("created_at"))
        if created is not None and created >= threshold:
            return str(run.get("id") or "unknown")
    return None


def _confirmed_dispatch_result(
    workflow_key,
    installation_token,
    *,
    environ=None,
    opener=None,
    dispatch_started,
):
    """Return an accepted result if GitHub recorded the dispatch, else None."""
    run_id = recent_dispatch_run_id(
        workflow_key,
        installation_token,
        environ=environ,
        opener=opener,
        since=dispatch_started,
    )
    if not run_id:
        return None
    key = str(workflow_key or "").strip()
    _log_event("github_dispatch_confirmed", workflow=key, run_id=run_id)
    return {"workflow": key, "status": "accepted", "run_id": run_id}


def dispatch_workflow(workflow_key, *, environ=None, opener=None, now=None):
    """Exchange App JWT, POST workflow dispatch, and return accepted result."""
    env = _env(environ)
    if opener is None:
        opener = urllib.request.urlopen
    installation_token = request_installation_token(environ=env, opener=opener, now=now)
    url, headers, body = build_dispatch_request(
        workflow_key, installation_token, environ=env
    )
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    dispatch_started = time.time()
    try:
        response = opener(request, timeout=_REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        status = _http_error_status(exc)
        close = getattr(exc, "close", None)
        if callable(close):
            close()
        if status in _NONRETRYABLE_HTTP_STATUSES:
            raise DispatchError(
                "github dispatch failed with status {}".format(status),
                retryable=False,
            )
        confirmed = _confirmed_dispatch_result(
            workflow_key,
            installation_token,
            environ=env,
            opener=opener,
            dispatch_started=dispatch_started,
        )
        if confirmed is not None:
            return confirmed
        raise DispatchError(
            "github dispatch failed with status {}".format(status),
            retryable=True,
        )
    except urllib.error.URLError:
        confirmed = _confirmed_dispatch_result(
            workflow_key,
            installation_token,
            environ=env,
            opener=opener,
            dispatch_started=dispatch_started,
        )
        if confirmed is not None:
            return confirmed
        raise DispatchError("github dispatch network error")
    except Exception:
        confirmed = _confirmed_dispatch_result(
            workflow_key,
            installation_token,
            environ=env,
            opener=opener,
            dispatch_started=dispatch_started,
        )
        if confirmed is not None:
            return confirmed
        raise DispatchError("github dispatch error")
    status = _read_status_and_close(response)
    try:
        status_code = int(status)
    except Exception:
        status_code = None
    if status_code is not None and 200 <= status_code < 300:
        key = str(workflow_key or "").strip()
        _log_event("github_dispatch_accepted", workflow=key, github_status=status_code)
        return {"workflow": key, "status": "accepted"}
    if status_code in _NONRETRYABLE_HTTP_STATUSES:
        raise DispatchError(
            "github dispatch returned status {}".format(status_code),
            retryable=False,
        )
    confirmed = _confirmed_dispatch_result(
        workflow_key,
        installation_token,
        environ=env,
        opener=opener,
        dispatch_started=dispatch_started,
    )
    if confirmed is not None:
        return confirmed
    raise DispatchError(
        "github dispatch returned status {}".format(status_code),
        retryable=True,
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "tradv2-dispatcher"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/dispatch":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._send_json(400, {"error": "invalid_payload"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"error": "invalid_payload"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "invalid_payload"})
            return
        workflow = payload.get("workflow")
        if workflow not in WORKFLOWS:
            self._send_json(400, {"error": "unsupported_workflow"})
            return
        try:
            result = dispatch_workflow(workflow)
        except DispatchError as exc:
            _log_event(
                "dispatch_request_failed",
                workflow=workflow,
                retryable=exc.retryable,
                reason=str(exc),
            )
            self._send_json(503 if exc.retryable else 502, {
                "error": "dispatch_failed",
                "workflow": workflow,
            })
            return
        self._send_json(202, result)


def main():
    try:
        port = int(os.environ.get("PORT") or 8080)
    except Exception:
        port = 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
