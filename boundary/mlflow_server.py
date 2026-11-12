"""The training dashboard's server: MLflow on musix, with a login, for runs on CSD3 to report to (``track``).

    python3 mlflow_server.py start      # the first start makes the passwords; later starts reuse them
    python3 mlflow_server.py status
    python3 mlflow_server.py stop

Everything lives in ``~/mlflow`` (``--dir``), which only you can read:
- ``auth.ini``: MLflow's login settings, with the admin password;
- ``secret_key``: the server's key for its login sessions;
- ``credentials``: the account the training jobs use (``boundary``), in MLflow's credentials format. Copy it to
  ``~/.mlflow/credentials`` on CSD3. It also signs you in to the dashboard;
- ``mlflow.db`` (runs), ``auth.db`` (accounts), ``server.log``, ``server.pid``.

The passwords are made at random and never printed. The server listens on every interface, because CSD3 has to
reach it, but:
- every request needs a login;
- it answers only to the host names in ``--allowed-hosts``;
- it accepts no file uploads (``--no-serve-artifacts``), where MLflow's worst past bugs were;
- it runs none of MLflow's background job workers.

Plain HTTP carries the login, so it crosses the University network unencrypted between CSD3 and musix. Viewing
goes through an SSH tunnel, which is encrypted.

Standard library only: run it with any Python 3. ``--mlflow`` names the MLflow to run, ``~/venvs/mlflow/bin/mlflow``
by default.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

JOBS_USER = "boundary"


def _private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def _ensure_secrets(d: Path) -> None:
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    if not (d / "auth.ini").exists():
        # auth.db is relative to the server's directory: SQLAlchemy reads the "@" in musix home paths as a login
        _private(d / "auth.ini", "[mlflow]\ndefault_permission = READ\ndatabase_uri = sqlite:///auth.db\n"
                 f"admin_username = admin\nadmin_password = {secrets.token_urlsafe(24)}\n"
                 "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n")
    if not (d / "secret_key").exists():
        _private(d / "secret_key", secrets.token_hex(32))
    if not (d / "credentials").exists():
        _private(d / "credentials", f"[mlflow]\nmlflow_tracking_username = {JOBS_USER}\n"
                 f"mlflow_tracking_password = {secrets.token_urlsafe(24)}\n")


def _ini(path: Path, key: str) -> str:
    for line in path.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    raise SystemExit(f"{path} has no {key}")


def _servers(port: int) -> list[int]:
    """This user's ``mlflow server`` processes on this port, however they were started, each with every process
    under it (the web server, and MLflow's job workers if it runs them). Not their process group: a server started
    without its own session shares one with the shell that started it."""
    out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,command="], capture_output=True, text=True)
    children, servers = {}, []
    for line in out.stdout.splitlines():
        pid, ppid, *args = line.split()
        children.setdefault(int(ppid), []).append(int(pid))
        if (any(a.endswith("/mlflow") or a == "mlflow" for a in args[:2]) and "server" in args
                and any(a == "--port" and b == str(port) for a, b in zip(args, args[1:]))):
            servers.append(int(pid))
    tree, todo = [], list(servers)
    while todo:
        pid = todo.pop()
        tree.append(pid)
        todo += children.get(pid, [])
    return tree


def _kill(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _request(base: str, method: str, path: str, login: tuple[str, str] | None = None, body: dict | None = None):
    headers = {"Content-Type": "application/json"}
    if login:
        headers["Authorization"] = "Basic " + base64.b64encode(":".join(login).encode()).decode()
    req = urllib.request.Request(base + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode()


def stop(args) -> None:
    pids = _servers(args.port)
    _kill(pids, signal.SIGTERM)
    for _ in range(30):
        if not any(_alive(p) for p in pids):
            break
        time.sleep(1)
    else:
        _kill(pids, signal.SIGKILL)
    (args.dir / "server.pid").unlink(missing_ok=True)
    print(f"stopped the server on port {args.port} ({len(pids)} processes)" if pids else f"no server on port {args.port}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def start(args) -> None:
    d = args.dir
    _ensure_secrets(d)
    if _servers(args.port):
        stop(args)
    hosts = f"{args.name}:{args.port},{args.name},localhost:*,127.0.0.1:*"
    env = dict(os.environ, MLFLOW_AUTH_CONFIG_PATH=str(d / "auth.ini"),
               MLFLOW_FLASK_SERVER_SECRET_KEY=(d / "secret_key").read_text().strip(),
               MLFLOW_SERVER_ENABLE_JOB_EXECUTION="false", MLFLOW_DISABLE_AGENT_HINT="1")
    with open(d / "server.log", "ab") as log:
        p = subprocess.Popen([str(args.mlflow), "server", "--app-name", "basic-auth", "--host", args.host,
                              "--port", str(args.port), "--workers", "1", "--backend-store-uri", "sqlite:///mlflow.db",
                              "--default-artifact-root", str(d / "artifacts"), "--no-serve-artifacts",
                              "--allowed-hosts", hosts],
                             cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    (d / "server.pid").write_text(f"{p.pid}\n")
    base = f"http://localhost:{args.port}"
    for _ in range(90):
        if p.poll() is not None:
            raise SystemExit(f"the server stopped at start; see {d / 'server.log'}")
        try:
            _request(base, "GET", "/health")
            break
        except urllib.error.HTTPError:          # answering, even if it wants a login for this too
            break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(f"no answer on port {args.port} after 90 s; see {d / 'server.log'}")
    admin = ("admin", _ini(d / "auth.ini", "admin_password"))
    jobs = (JOBS_USER, _ini(d / "credentials", "mlflow_tracking_password"))
    try:
        _request(base, "GET", f"/api/2.0/mlflow/users/get?username={JOBS_USER}", admin)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise SystemExit(f"checking the {JOBS_USER} account: HTTP {e.code}; see {d / 'server.log'}")
        _request(base, "POST", "/api/2.0/mlflow/users/create", admin, {"username": JOBS_USER, "password": jobs[1]})
    _request(base, "POST", "/api/2.0/mlflow/experiments/search", jobs, {"max_results": 1})   # the jobs login works
    try:
        _request(base, "POST", "/api/2.0/mlflow/experiments/search", None, {"max_results": 1})
        raise SystemExit("the server answered without a login; stopping it")
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
    print(f"MLflow on port {args.port} (pid {p.pid}), login required; the jobs account is in {d / 'credentials'}")


def status(args) -> None:
    pids = _servers(args.port)
    try:
        _request(f"http://localhost:{args.port}", "GET", "/health")
        health = "answering"
    except urllib.error.HTTPError:
        health = "answering"
    except OSError:
        health = "not answering"
    print(f"port {args.port}: {len(pids)} server processes, {health}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("start", "stop", "status"))
    ap.add_argument("--dir", type=Path, default=Path.home() / "mlflow")
    ap.add_argument("--mlflow", type=Path, default=Path.home() / "venvs/mlflow/bin/mlflow")
    ap.add_argument("--host", default="0.0.0.0", help="interface to listen on")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--name", default="musix.mus.cam.ac.uk", help="the host name jobs use to reach the server")
    args = ap.parse_args(argv)
    args.dir = args.dir.expanduser().resolve()
    {"start": start, "stop": stop, "status": status}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
