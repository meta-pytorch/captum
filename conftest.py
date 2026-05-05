"""Collection-time runner fingerprint for CI triage diagnostics.

Read-only probes only -- no credential bodies, no STS, no S3/ECR.
"""

import os
import sys
import urllib.error
import urllib.request

IMDS = "http://169.254.169.254"
TIMEOUT = 2.0


def _read_proc1_cgroup():
    with open("/proc/1/cgroup") as f:
        return f.readline().rstrip()


def _imds_v2_token():
    req = urllib.request.Request(
        f"{IMDS}/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        token = resp.read().decode().strip()
        return token or None


def _imds_get(path, token=None):
    headers = {}
    if token:
        headers["X-aws-ec2-metadata-token"] = token
    req = urllib.request.Request(f"{IMDS}{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode().strip()


def _safe(label, fn):
    try:
        return label, fn()
    except Exception as exc:
        return label, f"<error: {type(exc).__name__}: {exc}>"


def _gather():
    out = []
    out.append(_safe("hostname", lambda: os.uname().nodename))
    out.append(_safe("uname", lambda: " ".join(os.uname())))
    out.append(_safe("uid_gid", lambda: f"uid={os.getuid()} gid={os.getgid()}"))
    out.append(_safe("user_env", lambda: os.environ.get("USER", "<unset>")))
    out.append(_safe("cwd", lambda: os.getcwd()))
    out.append(_safe("proc1_cgroup", _read_proc1_cgroup))

    token = None
    mode = "none"
    try:
        token = _imds_v2_token()
        if token:
            mode = "v2"
    except Exception as exc:
        out.append(("imds_v2_error", f"{type(exc).__name__}: {exc}"))

    if mode == "none":
        try:
            _imds_get("/latest/meta-data/")
            mode = "v1"
        except Exception as exc:
            out.append(("imds_v1_error", f"{type(exc).__name__}: {exc}"))

    out.append(("imds_mode", mode))

    if mode != "none":
        out.append(_safe(
            "iam_role_name",
            lambda: _imds_get("/latest/meta-data/iam/security-credentials/", token),
        ))
        out.append(_safe(
            "az",
            lambda: _imds_get("/latest/meta-data/placement/availability-zone", token),
        ))
        out.append(_safe(
            "region",
            lambda: _imds_get("/latest/meta-data/placement/region", token),
        ))

        def _iid_prefix():
            iid = _imds_get("/latest/meta-data/instance-id", token)
            return f"{iid[:6]}...(len={len(iid)})"
        out.append(_safe("instance_id_prefix", _iid_prefix))

    return out


def _format(rows):
    lines = ["=" * 70, "RUNNER FINGERPRINT (research diagnostic)", "=" * 70]
    for label, value in rows:
        lines.append(f"  {label}: {value}")
    lines.append("=" * 70)
    return lines


try:
    _FINGERPRINT = _format(_gather())
except Exception as exc:
    _FINGERPRINT = [f"fingerprint_failed: {type(exc).__name__}: {exc}"]

try:
    for _line in _FINGERPRINT:
        print(_line, file=sys.stderr, flush=True)
except Exception:
    pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.section("RUNNER FINGERPRINT")
    for line in _FINGERPRINT:
        terminalreporter.write_line(line)
