#!/usr/bin/env python
"""Energy per document for the GPU-served embedding models, measured on the card.

The CPU sweep (scripts/bench_energy.py) covers the providers that run in this process. The models
worth using for quality do not: they are served by ollama on a GPU host, so the energy is drawn
somewhere else entirely and a probe on this box would measure only the HTTP client.

GPU power is a GAUGE, not a counter, so unlike RAPL it has to be integrated and the sampling rate
matters. Each window therefore gets its OWN sampler session - started before the work, stopped
after - rather than one long stream sliced by timestamps afterwards. Two hosts do not share a
clock, and slicing a remote stream with local window boundaries silently attributes whatever the
skew happens to be to the wrong side of each boundary.

Idle and work windows are interleaved for the same reason as the CPU sweep: the card is shared
with whatever else the node is running, and an idle baseline taken once absorbs it.

Env: MODELS (comma-separated ollama models), GPU_HOST, GPU_SSH_KEY (identity file for GPU_HOST;
unset uses ssh's own config and agent), DOCS (default 400), REPEATS (default 3),
IDLE_SECONDS (default 15), TARIFF (EUR/kWh, default 0.30), EMBED_BATCH (default 128), OUT.
"""

# pyright: basic
# One-off benchmark harness driving ssh probes; strict mode adds nothing over the shapes here.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _bench_energy import EnergyResult, cost_per_million, idle_estimate, integrate_power_w

# Optional: unset leaves the identity to ssh's own config and agent, which is the usual setup.
_KEY = os.environ.get("GPU_SSH_KEY", "")
# Written and read only on the REMOTE host, never opened locally, so the usual local-tempfile
# hazard does not apply. Named per pid regardless, so two concurrent sweeps cannot read each
# other's samples and silently attribute one model's power to another.
_REMOTE_CSV = f"/tmp/semdex-gpu-power-{os.getpid()}.csv"  # noqa: S108 - remote path, see above
DEFAULT_MODELS = ("qwen3-embedding:4b", "qwen3-embedding:8b", "bge-m3:latest")


def _ssh(host: str, command: str, *, timeout: float = 60.0) -> str:
    """One remote command. BatchMode so a rejected key fails fast instead of prompting."""
    result = subprocess.run(
        [
            "ssh",
            *(("-i", _KEY) if _KEY else ()),
            "-o",
            "BatchMode=yes",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=15",
            f"root@{host}",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=True,
    )
    return result.stdout


class GpuSampler:
    """Starts and stops an nvidia-smi power stream on the GPU host, one session per window."""

    def __init__(self, host: str, interval_ms: int = 200) -> None:
        self.host = host
        self.interval_ms = interval_ms

    def start(self) -> None:
        # setsid + redirect so the sampler outlives this ssh session rather than dying with it.
        self._stop_quietly()
        _ssh(
            self.host,
            f"setsid nohup nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits "
            f"-lms {self.interval_ms} > {_REMOTE_CSV} 2>/dev/null < /dev/null & echo started",
        )
        time.sleep(0.7)  # let the first samples land before the window opens

    def stop_and_read(self) -> list[tuple[float, float]]:
        """Stop sampling and return ``(seconds, watts)`` pairs.

        Timestamps are reconstructed from the sampler's own fixed interval rather than read from
        the rows: the value only has to be internally consistent for the integral, and deriving it
        from this host's clock would import the very skew the per-window design exists to avoid.
        """
        self._stop_quietly()
        raw = _ssh(self.host, f"cat {_REMOTE_CSV} 2>/dev/null || true")
        watts: list[float] = []
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                watts.append(float(text))
            except ValueError:  # a driver hiccup prints [N/A]; drop the row, keep the window
                continue
        step = self.interval_ms / 1000.0
        return [(i * step, w) for i, w in enumerate(watts)]

    def _stop_quietly(self) -> None:
        # pkill -x matches the executable name, never this command line - the whole ssh argv
        # contains "nvidia-smi", so a -f match would kill the shell issuing the kill.
        _ssh(self.host, "pkill -x nvidia-smi 2>/dev/null; true")

    def window(self, work: Any) -> tuple[float, float]:
        """Run ``work`` while sampling; return ``(joules, seconds)`` for the card."""
        self.start()
        started = time.perf_counter()
        work()
        elapsed = time.perf_counter() - started
        return integrate_power_w(self.stop_and_read()), elapsed

    def idle_watts(self, seconds: float) -> float:
        joules, elapsed = self.window(lambda: time.sleep(seconds))
        return joules / elapsed if elapsed else 0.0


def load_documents(n: int) -> list[str]:
    import ir_datasets

    out: list[str] = []
    for doc in ir_datasets.load("beir/nfcorpus/test").docs_iter():
        out.append((getattr(doc, "title", "") + " " + doc.text).strip())
        if len(out) >= n:
            break
    return out


def measure_model(model: str, documents: list[str], host: str, *, repeats: int, idle_seconds: float) -> dict[str, Any]:
    from semdex.composition import build_embedding
    from semdex.domain.enums import EmbeddingBackend

    embedding = build_embedding(
        EmbeddingBackend.OLLAMA,
        model=model,
        endpoint=f"http://{host}:11434",
        num_batch=4096,  # ollama truncates past its physical batch SILENTLY, at HTTP 200
        timeout=180.0,
        allow_fallback=False,
    )
    batch = int(os.environ.get("EMBED_BATCH", "128"))
    embedding.embed_passages(documents[:8])  # warm: the model load must not land inside a window

    def one_window() -> None:
        for i in range(0, len(documents), batch):
            embedding.embed_passages(documents[i : i + batch])

    sampler = GpuSampler(host)
    idles: list[float] = []
    work_j = 0.0
    work_s = 0.0
    for _ in range(repeats):
        idles.append(sampler.idle_watts(idle_seconds))
        joules, seconds = sampler.window(one_window)
        work_j += joules
        work_s += seconds
    idles.append(sampler.idle_watts(idle_seconds))

    idle_w, idle_spread = idle_estimate(idles)
    result = EnergyResult(
        total_j=work_j, seconds=work_s, idle_w=idle_w, idle_spread_w=idle_spread, units=len(documents) * repeats
    )
    tariff = float(os.environ.get("TARIFF", "0.30"))
    return {
        "provider": "ollama",
        "model_id": model,
        "dim": embedding.dim,
        "probe": "gpu-nvidia-smi",
        "documents": result.units,
        "mean_doc_bytes": round(sum(len(d.encode()) for d in documents) / len(documents), 1),
        "seconds": round(result.seconds, 2),
        "docs_per_s": round(result.units / result.seconds, 1) if result.seconds else 0.0,
        "average_w": round(result.average_w, 1),
        "idle_w": round(result.idle_w, 1),
        "idle_spread_w": round(result.idle_spread_w, 1),
        "marginal_j_per_doc": round(result.marginal_j_per_unit, 4),
        "total_j_per_doc": round(result.total_j_per_unit, 4),
        "signal_to_noise": round(result.signal_to_noise, 2),
        "resolved": result.resolved,
        "eur_per_million_docs_marginal": round(
            cost_per_million(result.marginal_j_per_unit, tariff_eur_per_kwh=tariff), 4
        ),
        "eur_per_million_docs_total": round(cost_per_million(result.total_j_per_unit, tariff_eur_per_kwh=tariff), 4),
    }


def _unload(host: str, model: str) -> None:
    """Evict the model from VRAM before the next one loads.

    Two embedding models resident on a 16 GB card thrash it, and the next model's row would then
    measure the eviction as well as its own work.
    """
    body = json.dumps({"model": model, "input": ["x"], "keep_alive": "0"})
    _ssh(host, f"curl -s localhost:11434/api/embed -d '{body}' >/dev/null; true")


def main() -> None:
    host = os.environ.get("GPU_HOST", "px-semdex-test-embeddings")
    docs = load_documents(int(os.environ.get("DOCS", "400")))
    repeats = int(os.environ.get("REPEATS", "3"))
    idle_seconds = float(os.environ.get("IDLE_SECONDS", "15"))
    models = os.environ.get("MODELS", ",".join(DEFAULT_MODELS)).split(",")
    card = _ssh(host, "nvidia-smi --query-gpu=name --format=csv,noheader").strip()
    print(f"{len(docs)} documents, {repeats} repeats, on {card} at {host}", flush=True)

    rows: list[dict[str, Any]] = []
    for model in models:
        print(f"model {model}", flush=True)
        try:
            row = measure_model(model, docs, host, repeats=repeats, idle_seconds=idle_seconds)
        except Exception as exc:  # one unavailable model must not end the sweep
            print(f"  SKIP {model}: {type(exc).__name__}: {exc}", flush=True)
            continue
        flag = "" if row["resolved"] else "  NOT RESOLVED"
        print(
            f"    {row['marginal_j_per_doc']} J/doc marginal, {row['docs_per_s']} docs/s, "
            f"idle {row['idle_w']}+/-{row['idle_spread_w']} W, s/n {row['signal_to_noise']}{flag}",
            flush=True,
        )
        rows.append(row)
        _unload(host, model)

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": host,
        "gpu": card,
        "tariff_eur_per_kwh": float(os.environ.get("TARIFF", "0.30")),
        "corpus": "beir/nfcorpus/test",
        "note": "Card power only: the host CPU serving the requests is not included, so these are "
        "a floor for a GPU deployment rather than its whole draw.",
        "rows": rows,
    }
    out_path = Path(os.environ.get("OUT", "energy-gpu.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
