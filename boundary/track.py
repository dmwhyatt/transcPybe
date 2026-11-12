"""Training progress sent to an MLflow server as it happens, so runs can be followed in a browser without logging in
to the cluster.

    with Tracker.start(out_dir, "r1/seed0", params, context_tags(out_dir)) as t:   # inert without MLFLOW_TRACKING_URI
        ...
        t.log(record, step)                                                         # one request per evaluation

The server is the one the CMS guide runs on musix, whose firewall opens port 5000 to CSD3; ``mlflow_server.py``
starts it there with a login. The job signs in as MLflow's own clients do, from MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD or else ``~/.mlflow/credentials``. Runs are grouped under the experiment in
MLFLOW_EXPERIMENT_NAME, ``boundary`` by default. Only the standard library is used, through
MLflow's REST API, so a job's environment needs nothing new. The run id is kept in ``RUN_DIR/mlflow_run_id``, so a
resumed run carries on in the same MLflow run.

The server is never allowed to slow or stop training. Each request waits at most 5 s, and after three failures in
a row the run stops reporting, with one line in its log saying so. ``log.jsonl`` in the run directory stays the
record; the dashboard is only a view of it.
"""
from __future__ import annotations

import base64
import configparser
import json
import math
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIMEOUT_S = 5.0
MAX_FAILURES = 3
PARAMS_PER_REQUEST = 100     # MLflow's limit for one log-batch request
PARAM_CHARS = 500            # long values are cut short here; args.json in the run directory keeps them whole


def _now_ms() -> int:
    return int(time.time() * 1000)


def _say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [track] {msg}", flush=True)


def flatten(record: dict, prefix: str = "") -> dict:
    """Finite numbers only, with nested sets named ``extra.NAME.onset_f``; MLflow rejects NaN in JSON."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[prefix + k] = float(v)
    return out


def credentials() -> tuple[str, str] | None:
    """Username and password from where MLflow's clients look: the environment first, then ~/.mlflow/credentials."""
    user, password = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
    path = Path.home() / ".mlflow" / "credentials"
    if not (user and password) and path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(path)
        section = cfg["mlflow"] if cfg.has_section("mlflow") else {}
        user = user or section.get("mlflow_tracking_username")
        password = password or section.get("mlflow_tracking_password")
    return (user, password) if user and password else None


def context_tags(out: Path) -> dict:
    """Where a run is happening: its Slurm job, node, run directory, user and commit."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID")
    job = f"{job}_{os.environ.get('SLURM_ARRAY_TASK_ID')}" if job else os.environ.get("SLURM_JOB_ID")
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"slurm_job": job, "node": socket.gethostname().split(".")[0], "run_dir": str(out),
            "mlflow.user": os.environ.get("USER"), "mlflow.source.git.commit": commit}


class Tracker:
    def __init__(self, uri: str | None):
        self.uri = uri.rstrip("/") if uri else None
        self.run_id: str | None = None
        self.failures = 0
        login = credentials() if uri else None
        self.headers = {"Content-Type": "application/json"}
        if login:
            self.headers["Authorization"] = "Basic " + base64.b64encode(":".join(login).encode()).decode()

    def _call(self, method: str, endpoint: str, body: dict | None = None, query: dict | None = None,
              expected: tuple = ()) -> dict | None:
        """The parsed reply, or None. An HTTP status in ``expected`` is an answer, not a failure."""
        if self.uri is None:
            return None
        url = f"{self.uri}/api/2.0/mlflow/{endpoint}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        req = urllib.request.Request(url, method=method, headers=self.headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                self.failures = 0
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in expected:
                return None
            err = f"HTTP {e.code} {e.read()[:200].decode(errors='replace')}"
            if e.code in (401, 403):
                err += (" (no login found: put the jobs account in ~/.mlflow/credentials)"
                        if "Authorization" not in self.headers else " (the login was refused)")
        except (OSError, ValueError) as e:          # refused, unreachable, timed out, or not JSON
            err = str(e)
        self.failures += 1
        _say(f"{endpoint}: {err}")
        if self.failures >= MAX_FAILURES:
            _say(f"{self.uri} failed {MAX_FAILURES} times in a row; this run stops reporting to it")
            self.uri = None
        return None

    def _experiment(self, name: str) -> str | None:
        # two seeds starting side by side may both find it missing; the second create is refused, so look again
        for _ in range(2):
            got = self._call("GET", "experiments/get-by-name", query={"experiment_name": name}, expected=(404,))
            if got:
                return got["experiment"]["experiment_id"]
            made = self._call("POST", "experiments/create", {"name": name}, expected=(400,))
            if made:
                return made["experiment_id"]
        return None

    @classmethod
    def start(cls, out: Path, name: str, params: dict, tags: dict) -> "Tracker":
        t = cls(os.environ.get("MLFLOW_TRACKING_URI") or None)
        if t.uri is None:
            return t
        if not t.uri.startswith(("http://", "https://")):
            _say(f"MLFLOW_TRACKING_URI must be an http(s) server, not {t.uri}; not reporting")
            t.uri = None
            return t
        id_file = out / "mlflow_run_id"
        if id_file.exists():                        # a resumed run: same MLflow run, parameters already there
            t.run_id = id_file.read_text().strip()
            if t._call("POST", "runs/update", {"run_id": t.run_id, "status": "RUNNING"}) is not None:
                _say(f"resuming MLflow run {t.run_id} on {t.uri}")
            return t
        eid = t._experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or "boundary")
        run = eid and t._call("POST", "runs/create", {
            "experiment_id": eid, "run_name": name, "start_time": _now_ms(),
            "tags": [{"key": k, "value": str(v)} for k, v in tags.items() if v not in (None, "")]})
        if not run:
            return t
        t.run_id = run["run"]["info"]["run_id"]
        id_file.write_text(t.run_id + "\n")
        items = [{"key": k, "value": str(v)[:PARAM_CHARS]} for k, v in params.items()]
        for i in range(0, len(items), PARAMS_PER_REQUEST):
            t._call("POST", "runs/log-batch", {"run_id": t.run_id, "params": items[i:i + PARAMS_PER_REQUEST]})
        _say(f"reporting to {t.uri} as run {t.run_id} ({name})")
        return t

    def log(self, record: dict, step: int) -> None:
        if self.run_id is None:
            return
        ts = _now_ms()
        metrics = [{"key": k, "value": v, "timestamp": ts, "step": step} for k, v in flatten(record).items()]
        self._call("POST", "runs/log-batch", {"run_id": self.run_id, "metrics": metrics})

    def end(self, status: str = "FINISHED") -> None:
        if self.run_id is not None:
            self._call("POST", "runs/update", {"run_id": self.run_id, "status": status, "end_time": _now_ms()})

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, kind, err, tb) -> None:
        """Marks the run FINISHED, KILLED (stopped: a time limit's SIGTERM, or Ctrl-C) or FAILED."""
        stopped = kind is not None and issubclass(kind, (KeyboardInterrupt, SystemExit))
        self.end("FINISHED" if kind is None else "KILLED" if stopped else "FAILED")
