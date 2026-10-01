#!/usr/bin/env python
"""Health check for a running pre-embed / breakpoint sweep and (optionally) a remote GPU host.

A companion to preembed_vectors.py / sweep_breakpoint_models.py: it reads only DURABLE signals so
it keeps working across restarts - the cache on disk (CACHE_ROOT), local /proc, and an OPTIONAL
remote GPU host reached over SSH. Prints OK / PROBLEM lines; exit 0 if all OK, 1 if any PROBLEM
(so a cron or the watchdog can branch on it).

All infra specifics are env-injected - the defaults touch nothing remote, so a bare run only checks
the local cache and processes. Config (scripts/ convention - no argparse):

  CACHE_ROOT                    cache dir to watch (default /embeddings)
  SEMDEX_HEALTH_STATE           progress-marker state file for stall detection
                                (default <cache>/scores/sweep_health_state.json)
  SEMDEX_WATCHDOG_SSH           SSH command prefix to the GPU host, shell-split
                                (e.g. "ssh -i KEY -o BatchMode=yes root@gpuhost"). EMPTY = skip all
                                remote GPU/ollama checks (local-only run).
  SEMDEX_WATCHDOG_OLLAMA_EXEC   command prefix run ON the SSH host to reach the ollama host
                                (e.g. "pct exec 60400 --" for an LXC guest); default "" (direct).
  SEMDEX_WATCHDOG_OLLAMA_URL    ollama base URL as seen from the exec target (default
                                http://127.0.0.1:11434)
  SEMDEX_WATCHDOG_OLLAMA_MODEL  ENABLE FLAG for the liveness probe; EMPTY = skip the probe. Any
                                non-empty value turns it on. The model is NOT taken from here: the
                                probe embeds whatever /api/ps reports as already loaded, and falls
                                back to a load-free /api/tags call when ollama is idle. Naming a
                                model here would make the probe LOAD it, evicting the model the
                                sweep is using (see _resident_model).
  SEMDEX_WATCHDOG_OLLAMA_TIMEOUT seconds for the liveness embed probe (default 30). Must be
                                generous: under 100% GPU load the tiny probe queues behind the
                                current embed batch, so a tight timeout (8s) false-alarms "wedge".
                                The probe also retries once before flagging.
  SEMDEX_WATCHDOG_PROTECT_PID   optional PID that MUST stay alive (a long CPU sweep you must not
                                lose); PROBLEM if it vanishes. Default: none.
  SEMDEX_WATCHDOG_MIN_FREE_GB   warn-below free space on CACHE_ROOT (default 50)
  SEMDEX_WATCHDOG_EMBED_STALL_S / _CHUNK_STALL_S  seconds of no .tmp growth before "stale"
                                (defaults 1200 / 2400)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import TypedDict, cast


class _State(TypedDict, total=False):
    """The stall-detection marker file: last seen sizes plus when they were seen."""

    embed_sz: int
    chunk_sz: int
    ts: float


CACHE = Path(os.environ.get("CACHE_ROOT", "/embeddings"))
STATE = Path(os.environ.get("SEMDEX_HEALTH_STATE", str(CACHE / "scores" / "sweep_health_state.json")))
SSH = shlex.split(os.environ.get("SEMDEX_WATCHDOG_SSH", ""))
OLLAMA_EXEC = shlex.split(os.environ.get("SEMDEX_WATCHDOG_OLLAMA_EXEC", ""))
OLLAMA_URL = os.environ.get("SEMDEX_WATCHDOG_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SEMDEX_WATCHDOG_OLLAMA_MODEL", "")
OLLAMA_TIMEOUT = int(os.environ.get("SEMDEX_WATCHDOG_OLLAMA_TIMEOUT", "30"))
PROTECT_PID = os.environ.get("SEMDEX_WATCHDOG_PROTECT_PID", "")
MIN_FREE_GB = int(os.environ.get("SEMDEX_WATCHDOG_MIN_FREE_GB", "50"))
EMBED_STALL_S = int(os.environ.get("SEMDEX_WATCHDOG_EMBED_STALL_S", "1200"))
CHUNK_STALL_S = int(os.environ.get("SEMDEX_WATCHDOG_CHUNK_STALL_S", "2400"))

problems: list[str] = []
oks: list[str] = []


def _sh(cmd: list[str], timeout: int = 30) -> str:
    try:
        # check=False: a non-zero probe is a signal we classify below, never an exception here.
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except Exception as e:
        return f"__ERR__ {type(e).__name__} {e}"


def _newest_tmp(pattern: str) -> tuple[str, float, int]:
    """(dir-name, mtime, size) of the newest matching *.tmp - a phase's live write."""
    name, mt, sz = "", 0.0, 0
    # Path.glob() needs a root plus a RELATIVE pattern; the callers pass absolute paths, so split
    # the anchor off rather than stripping "/" (which would silently glob the cwd instead).
    anchor = Path(pattern).anchor
    for p in Path(anchor or ".").glob(pattern[len(anchor) :] if anchor else pattern):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime > mt:
            name, mt, sz = p.parent.name, st.st_mtime, st.st_size
    return name, mt, sz


def _proc_count(needle: str) -> int:
    n = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            c = (entry / "cmdline").read_bytes().decode("utf8", "replace")
        except OSError:
            continue
        if needle in c:
            n += 1
    return n


def _check_cache(now: float, prev: _State) -> tuple[int, int, float, bool]:
    """(embed size, chunk size, embed mtime, embed_live) - embed_live is returned, not global.

    embed_live is ground truth for the GPU/ollama checks: if the embed cache is advancing then the
    remote GPU and ollama ARE working, so a timed-out SSH probe means the node is momentarily too
    loaded to answer, not a wedge.
    """
    embed_live = False
    en, emt, esz = _newest_tmp(str(CACHE / "vectors/*/vectors.npy.tmp"))
    if emt == 0.0:
        oks.append("embed: no active vectors.tmp (between cells)")
    elif (now - emt) < EMBED_STALL_S or esz != prev.get("embed_sz", -1):
        embed_live = True
        oks.append(f"embed advancing: {en} ({int(now - emt)}s ago, {esz // 1024}KB)")
    else:
        problems.append(f"embed vectors.tmp {en} STALE ({int(now - emt)}s, no growth) - wedge?")

    cn, cmt, csz = _newest_tmp(str(CACHE / "chunks/*/chunks.parquet.tmp"))
    if cmt == 0.0:
        oks.append("chunk: no active .tmp (sets may be complete)")
    elif (now - cmt) < CHUNK_STALL_S or csz != prev.get("chunk_sz", -1):
        oks.append(f"chunk writing: {cn} ({int(now - cmt)}s ago)")
    else:
        problems.append(f"chunk .tmp {cn} STALE ({int(now - cmt)}s) - chunk stall?")
    return esz, csz, emt, embed_live


def _check_procs(now: float, emt: float) -> None:
    sweeps = _proc_count("sweep_breakpoint_models.py") + _proc_count("preembed_vectors.py")
    if sweeps == 0 and emt and (now - emt) > 300:
        problems.append("NO sweep/preembed process alive AND embed not writing")
    else:
        oks.append(f"{sweeps} sweep/preembed process(es) alive")
    if PROTECT_PID:
        if Path(f"/proc/{PROTECT_PID}").exists():
            oks.append(f"protected PID {PROTECT_PID} alive")
        else:
            problems.append(f"protected PID {PROTECT_PID} GONE")


def _check_gpu(*, embed_live: bool) -> None:
    if not SSH:
        return
    gpu = _sh(
        [
            *SSH,
            "for i in 1 2 3 4 5; do nvidia-smi "
            "--query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 0.3; done",
        ]
    )
    lines = gpu.splitlines()
    utils = [int(ln.split("%")[0]) for ln in lines if "%" in ln and ln.split("%")[0].strip().isdigit()]
    vram = [int(ln.split(",")[1].strip().split()[0]) for ln in lines if "," in ln and "MiB" in ln]
    if not utils:
        msg = f"GPU query failed (SSH timed out): {gpu[:60]}"
        oks.append(f"{msg} - but embed advancing, node just loaded") if embed_live else problems.append(msg)
    elif max(utils) == 0:
        msg = f"GPU IDLE (all {len(utils)} samples 0%)"
        oks.append(f"{msg} - but embed advancing (probe artifact)") if embed_live else problems.append(
            f"{msg} - embed lane starved / wedge"
        )
    else:
        oks.append(f"GPU util {min(utils)}-{max(utils)}%, VRAM {max(vram) if vram else '?'}MiB")
    if vram and max(vram) > 15900:
        problems.append(f"VRAM {max(vram)}MiB near cap - thrash risk")


def _ollama_curl(path: str, payload: str | None = None, *, body: bool = False) -> str:
    """Call an ollama endpoint through the SSH+exec chain; HTTP status, or the body if body=True."""
    remote = [*OLLAMA_EXEC, "curl", "-s", "-m", str(OLLAMA_TIMEOUT)]
    if not body:
        remote += ["-o", "/dev/null", "-w", "%{http_code}"]
    remote += [f"{OLLAMA_URL}{path}"]
    if payload is not None:
        remote += ["-d", payload]
    return _sh([*SSH, shlex.join(remote)], timeout=OLLAMA_TIMEOUT + 10)


def _resident_model() -> str:
    """Name of a model ollama ALREADY has loaded, or '' when it is idle.

    The probe must never name a model of its own choosing. POSTing /api/embed with a model that
    is not resident makes ollama LOAD it, and on a GPU sized for one model that EVICTS the model
    the sweep is using. Measured 2026-07-28: a probe pinned to qwen3-embedding:4b, firing every
    30 min, evicted the live qwen3-embedding:0.6b breakpoint model; it reloaded at 0/29 layers on
    GPU and the chunk phase ran ~4x slower (embed median 13.8s vs 4.7s) with no error anywhere -
    every component reported healthy while the probe itself was causing the damage.
    """
    raw = _ollama_curl("/api/ps", body=True)
    try:
        parsed: object = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    loaded: object = cast("dict[str, object]", parsed).get("models")
    if not isinstance(loaded, list):
        return ""
    for entry in cast("list[object]", loaded):
        if not isinstance(entry, dict):
            continue
        fields = cast("dict[str, object]", entry)
        name: object = fields.get("model") or fields.get("name")
        if isinstance(name, str) and name:
            return name
    return ""


def _check_ollama(*, embed_live: bool) -> None:
    if not SSH or not OLLAMA_MODEL:
        return
    model = _resident_model()
    if model:
        # Real compute proof, at zero VRAM cost: exercise whatever is already loaded.
        target, payload, what = "/api/embed", json.dumps({"model": model, "input": "x"}), f"embed on resident {model}"
    else:
        # Idle: liveness only. Loading a model just to answer the probe is the bug above.
        target, payload, what = "/api/tags", None, "tags (no model resident)"
    # retry once: under full GPU load the probe queues behind the current batch and a single slow
    # tick is not a wedge (verified: a 30s-timeout probe answers in ~0s while an 8s one times out).
    code = ""
    for _ in range(2):
        code = _ollama_curl(target, payload)
        if code.endswith("200"):
            break
    if code.endswith("200"):
        oks.append(f"ollama responsive (http 200, {what})")
    elif embed_live:
        # the embed cache is advancing => ollama IS serving; the probe just lost the race to node load
        oks.append(f"ollama probe timed out (rc={code!r}) - but embed advancing, so ollama is alive")
    else:
        problems.append(f"ollama NOT healthy (rc={code!r}) - possible wedge")


def _check_disk() -> None:
    df = _sh(["df", "-BG", "--output=avail", str(CACHE)])
    availg = df.splitlines()[-1].strip().rstrip("G") if df and "__ERR__" not in df else ""
    if availg.isdigit():
        oks.append(f"{CACHE} free: {availg}G")
        if int(availg) < MIN_FREE_GB:
            problems.append(f"{CACHE} LOW space: {availg}G (< {MIN_FREE_GB}G)")


def main() -> int:
    now = time.time()
    prev: _State = cast("_State", json.loads(STATE.read_text())) if STATE.exists() else _State()
    esz, csz, emt, embed_live = _check_cache(now, prev)
    _check_procs(now, emt)
    _check_gpu(embed_live=embed_live)
    _check_ollama(embed_live=embed_live)
    _check_disk()

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"embed_sz": esz, "chunk_sz": csz, "ts": now}))

    print(f"=== sweep health @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    for o in oks:
        print(f"  OK   {o}")
    for p in problems:
        print(f"  !!   PROBLEM: {p}")
    print(f"=== verdict: {'PROBLEM x' + str(len(problems)) if problems else 'ALL OK'} ===")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
