#!/usr/bin/env python
"""Standalone e-mail watchdog around bp_sweep_healthcheck.py - survives its launching session.

Runs detached (nohup/setsid). Every INTERVAL it runs bp_sweep_healthcheck.py and e-mails via
`semdex send-notification` ONLY on a state transition or a long-running problem, so no per-tick
spam:

  OK  -> PROBLEM : alert immediately, with the failing lines.
  PROBLEM (still): re-alert every REALERT seconds.
  PROBLEM -> OK  : one "recovered" mail.

E-mail-only, NO corrective action by design: if a supervising session is alive it fixes things;
two auto-fixers would fight. Every tick is logged. Point it at a durable location (not a volatile
tmpdir) so it outlives the session. Config (scripts/ convention):

  all SEMDEX_WATCHDOG_* / CACHE_ROOT vars of bp_sweep_healthcheck.py (passed through), plus:
  SEMDEX_WATCHDOG_INTERVAL_S    seconds between checks (default 1800 = 30 min)
  SEMDEX_WATCHDOG_REALERT_S     re-mail a still-broken problem every N seconds (default 10800 = 3h)
  SEMDEX_WATCHDOG_LOG           tick log path (default <dir>/watchdog.log)
  SEMDEX_WATCHDOG_SEMDEX_BIN    semdex CLI for send-notification (default: semdex on PATH)
  SEMDEX_WATCHDOG_SEMDEX_DIR    cwd for the CLI (.env discovery); default the repo root
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
HEALTHCHECK = HERE / "bp_sweep_healthcheck.py"
PY = os.environ.get("SEMDEX_WATCHDOG_PYTHON", "python3")
LOG = Path(os.environ.get("SEMDEX_WATCHDOG_LOG", str(HERE / "watchdog.log")))
SEMDEX_BIN = os.environ.get("SEMDEX_WATCHDOG_SEMDEX_BIN", "semdex")
SEMDEX_DIR = os.environ.get("SEMDEX_WATCHDOG_SEMDEX_DIR", str(REPO_ROOT))
INTERVAL = int(os.environ.get("SEMDEX_WATCHDOG_INTERVAL_S", "1800"))
REALERT = int(os.environ.get("SEMDEX_WATCHDOG_REALERT_S", str(3 * 3600)))


def _log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def _email(subject: str, message: str) -> None:
    try:
        r = subprocess.run(
            [SEMDEX_BIN, "send-notification", "--subject", subject, "--message", message],
            cwd=SEMDEX_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        _log(f"email '{subject}' rc={r.returncode}" + ("" if r.returncode == 0 else f" ERR {r.stderr.strip()[:200]}"))
    except Exception as e:
        _log(f"email EXC {type(e).__name__} {e}")


def _run_check() -> tuple[int, str]:
    try:
        # check=False: the healthcheck signals PROBLEM with exit 1, which we classify, not raise on.
        r = subprocess.run([PY, str(HEALTHCHECK)], capture_output=True, text=True, timeout=180, check=False)
        return r.returncode, r.stdout.strip()
    except Exception as e:
        return 2, f"healthcheck itself failed: {type(e).__name__} {e}"


def main() -> None:
    _log(f"watchdog START (interval={INTERVAL}s, realert={REALERT}s)")
    _email(
        "[semdex-watchdog] gestartet",
        f"Der bp-Sweep-Watchdog laeuft (alle {INTERVAL // 60} min, E-Mail nur bei Problemen). "
        f"Er ueberlebt ein Ende der launcherenden Session. Log: {LOG}",
    )
    prev_problem = False
    last_alert = 0.0
    while True:
        rc, out = _run_check()
        problem = rc != 0
        now = time.time()
        _log(out.splitlines()[-1] if out else f"rc={rc}")
        if problem and not prev_problem:
            _email("[semdex-watchdog] PROBLEM erkannt", out)
            last_alert = now
        elif problem and (now - last_alert) >= REALERT:
            _email("[semdex-watchdog] PROBLEM haelt an", out)
            last_alert = now
        elif prev_problem and not problem:
            _email("[semdex-watchdog] wieder OK", out)
        prev_problem = problem
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
