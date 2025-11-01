#!/usr/bin/env python3
"""
make_corpus.py — build /data/tgdk_corpus/*.jsonl for DAPT

Creates:
  tgdk_docs.jsonl
  math_book.jsonl
  code_repos.jsonl
  strategy_playbooks.jsonl
  sat_ops.jsonl
  dialogs_seed.jsonl

Each line: {"text": "..."}
"""

import os, re, json, hashlib, argparse, textwrap
from pathlib import Path

# -------- utils --------

EXCLUDES_DIR = {
    ".git", ".svn", ".hg", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox",
    "node_modules", "dist", "build", "venv", ".venv", ".idea", ".vscode",
    "seals", "certificates", "certs", "coverage", "site-packages", ".cache",
    "mission-ui/dist", "frontend/dist"
}

CODE_EXTS = {
    ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".rst", ".txt",
    ".yml", ".yaml", ".ini", ".cfg", ".toml", ".sql",
    ".sh", ".bash", ".ps1", ".bat",
    ".html", ".css", ".scss",
    ".cpp", ".cc", ".cxx", ".c", ".hpp", ".h", ".cu",
    ".cmake", ".dockerignore", ".gitattributes", ".gitignore",
}
MAX_FILE_BYTES = 500_000  # skip huge blobs

def is_probably_text(p: Path) -> bool:
    if p.name == "Dockerfile": return True
    return p.suffix in CODE_EXTS

def chunk(text: str, max_chars: int = 4000):
    i = 0
    n = len(text)
    while i < n:
        yield text[i:i+max_chars]
        i += max_chars

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")

def dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        h = hashlib.sha1(r.encode("utf-8", "ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    return out

# -------- generators --------

def gen_code_repos(root: Path):
    rows = []
    for p in root.rglob("*"):
        if p.is_dir():
            # skip excluded dirs anywhere in path
            parts = set(p.parts)
            if parts & EXCLUDES_DIR:
                continue
            continue
        parts = set(p.parts)
        if parts & EXCLUDES_DIR:
            continue
        if p.name == "Dockerfile" or is_probably_text(p):
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(p.relative_to(root))
            header = f"FILE: {rel}\n---\n"
            for ch in chunk(content, 3500):
                rows.append(header + ch)
    return rows

def gen_tgdk_docs():
    rows = []

    bfe = textwrap.dedent("""
    TGDK OPEN SOURCE LICENSE (TSOL) — BFE (Broadcasted Fee Entry)
    © 2025 TGDK & OliviaAI. All Rights Reserved.

    Summary:
    - Use, modify, redistribute with BFE compliance.
    - Security Framework: TichenorCode (T-Code).
    - Attribution: Preserve TGDK/Olivia notices in UI/docs.
    - Liability: Provided "AS IS" without warranties.
    """).strip()

    tcode = textwrap.dedent("""
    T-Code (TichenorCode) Overview

    Sealed key files:
      - ./seals/t_code.seal        (binary sealed key)
      - ./seals/t_code.current     (pointer file, contains KID)
    Environment exports:
      - T_CODE_KID, T_CODE_KEY (base64url; no-padding)
    Rotation policy:
      - max_age_s env-driven; rotate only on schedule or manual force.
    """).strip()

    api = textwrap.dedent("""
    QuomoSatNet API — Overview

    Services:
      - web (FastAPI): mission control, uplink, certificates, scoring
      - worker (Celery): async tasks
      - redis: broker/result backend
      - db (Postgres): satellites, tasks, logs
      - telenet (test shim): TCP echo/socket harness
      - mission-ui (nginx): static dashboard

    Key Routers:
      - /certificates
      - /scoring
      - /propagation
      - /stream
      - /orders/process
    """).strip()

    rows += [bfe, tcode, api]
    return rows

def gen_math_book():
    rows = []

    fspl = textwrap.dedent(r"""
    Free-Space Path Loss (FSPL)
    FSPL[dB] = 20·log10(d_km) + 20·log10(f_MHz) + 32.44

    Example:
      d = 35,786 km (GEO)
      f = 2250 MHz
      FSPL ≈ 20*log10(35786) + 20*log10(2250) + 32.44 ≈ 196.6 dB
    """).strip()

    eirp = textwrap.dedent(r"""
    EIRP (Equivalent Isotropically Radiated Power)
    P[dBW] = 10·log10(P_watts)
    EIRP[dBW] = P[dBW] + G_tx[dBi] - L_tx[dB]

    Example:
      P = 50 W → P[dBW] = 10·log10(50) ≈ 16.99 dBW
      G_tx = 30 dBi, L_tx = 0.5 dB
      EIRP ≈ 16.99 + 30 - 0.5 = 46.49 dBW
    """).strip()

    cn0 = textwrap.dedent(r"""
    Carrier to Noise Density, C/N0
    C/N0[dB-Hz] = EIRP - FSPL + G_rx[dBi] - (10·log10(k·T_sys))
    where k = 1.38e-23 J/K, T_sys in Kelvin

    Example:
      EIRP = 46.5 dBW, FSPL = 196.6 dB
      G_rx = 40 dBi, T_sys = 290 K
      10·log10(kT) ≈ 10·log10(1.38e-23*290) ≈ -198.6 dBW/Hz
      C/N0 ≈ 46.5 - 196.6 + 40 - (-198.6) ≈ 88.5 dB-Hz
    """).strip()

    ebno = textwrap.dedent(r"""
    Energy per bit to Noise density, Eb/N0
    Eb/N0[dB] = C/N0[dB-Hz] - 10·log10(R_bps)

    Example:
      C/N0 = 88.5 dB-Hz, R = 1e6 bps → 10·log10(1e6)=60 dB
      Eb/N0 ≈ 88.5 - 60 = 28.5 dB
    """).strip()

    sanity = textwrap.dedent(r"""
    Sanity checks:
      - Units consistent (MHz vs Hz, km vs m)
      - Antenna gains in dBi, losses in dB, power in dBW
      - GEO slant range ~ 35,786 km baseline
      - BER vs Eb/N0 curves depend on modulation/coding
    """).strip()

    rows += [fspl, eirp, cn0, ebno, sanity]
    return rows

def gen_strategy_playbooks():
    rows = []

    runbook1 = """
    Incident: DB password desync (InvalidPasswordError for user "satnet")

    Symptoms:
      - app logs: asyncpg.exceptions.InvalidPasswordError
      - health returns degraded or 500 when touching DB

    Actions:
      1) Ensure consistent ENV in compose (POSTGRES_USER/DB/PASSWORD)
      2) If rotated, re-sync: psql -U postgres -c "ALTER USER satnet WITH PASSWORD 'password';"
      3) Restart dependent services: web, worker
      4) Confirm with: docker exec web python -c 'import asyncpg; ... connect ...'
    """

    runbook2 = """
    Incident: Telenet unreachable / connect timed out

    Actions:
      1) Confirm service up: docker compose ps telenet
      2) From web: socket.create_connection(('telenet', 9200), 2) should succeed
      3) Disable legacy blocking probes and run async loop only
      4) Healthcheck: expose port 9200 and simple echo (socat)
    """

    runbook3 = """
    Incident: Native module tgdk_bindings missing

    Actions:
      1) Build C++ with cmake; install to /out
      2) At runtime, copy tgdk_bindings*.so into site-packages OR use pure-Python shim
      3) Ensure LD_LIBRARY_PATH includes /out/lib
    """

    rows += [textwrap.dedent(runbook1), textwrap.dedent(runbook2), textwrap.dedent(runbook3)]
    return rows

def gen_sat_ops():
    rows = []

    story = textwrap.dedent("""
    SAT OPS Narrative: Uplink budget check for SAT-041 (X-band)

    Assumptions:
      f = 8.2 GHz (8200 MHz), d = 1200 km (LEO pass), P_tx = 20 W,
      G_tx = 25 dBi, L_tx = 1.0 dB, G_rx = 38 dBi, R = 2.0e6 bps, T_sys = 350 K.

    Steps:
      1) FSPL = 20log10(1200) + 20log10(8200) + 32.44
      2) EIRP = 10log10(20) + 25 - 1
      3) C/N0 = EIRP - FSPL + G_rx - 10log10(kT)
      4) Eb/N0 = C/N0 - 10log10(R)

    Validate against expected thresholds for chosen modulation.
    """).strip()

    example_json = {
        "sat_id": "SAT-041",
        "link": {
            "freq_mhz": 8200.0,
            "range_km": 1200.0,
            "tx_power_w": 20.0,
            "tx_gain_dbi": 25.0,
            "tx_loss_db": 1.0,
            "rx_gain_dbi": 38.0,
            "bitrate_bps": 2_000_000.0,
            "tsys_k": 350.0
        },
        "expected": {
            "eb_no_db": 8.0,
            "c_to_n0_dbhz": 75.0
        }
    }

    rows.append(story)
    rows.append("SAT OPS JSON EXAMPLE:\n" + json.dumps(example_json, indent=2))
    return rows

def gen_dialogs_seed():
    rows = []

    dialogs = [
        textwrap.dedent("""
        System: You are Olivia, TGDK ops assistant.
        User: Generate BFE licenses for SAT-041 and SAT-042 as a zip.
        Assistant: Creating two BFE LICENSE files with TSOL/BFE headers, signing with T-Code KID, bundling to /documents/quomo/<date>_SATS.zip (server-side)."""),
        textwrap.dedent("""
        System: You are Olivia, TGDK ops assistant.
        User: Our health endpoint is degraded: DB password invalid.
        Assistant: Re-sync Postgres user password for 'satnet' to env, restart web/worker, confirm with asyncpg one-liner, then re-run migrations."""),
        textwrap.dedent("""
        System: You are Olivia, TGDK ops assistant.
        User: Compute EB/N0 given C/N0=88.5 dB-Hz and R=1e6.
        Assistant: Eb/N0 = 88.5 - 10·log10(1e6) = 28.5 dB."""),
    ]

    rows += dialogs
    return rows

# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/app", help="repo root to crawl for code")
    ap.add_argument("--out",  default="/data/tgdk_corpus", help="output dir")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out  = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Build
    code_rows = dedupe(gen_code_repos(root))
    docs_rows = dedupe(gen_tgdk_docs())
    math_rows = dedupe(gen_math_book())
    strat_rows = dedupe(gen_strategy_playbooks())
    sat_rows = dedupe(gen_sat_ops())
    dialog_rows = dedupe(gen_dialogs_seed())

    write_jsonl(out / "code_repos.jsonl", code_rows)
    write_jsonl(out / "tgdk_docs.jsonl", docs_rows)
    write_jsonl(out / "math_book.jsonl", math_rows)
    write_jsonl(out / "strategy_playbooks.jsonl", strat_rows)
    write_jsonl(out / "sat_ops.jsonl", sat_rows)
    write_jsonl(out / "dialogs_seed.jsonl", dialog_rows)

    # Summary
    print(json.dumps({
        "out_dir": str(out),
        "counts": {
            "code_repos": len(code_rows),
            "tgdk_docs": len(docs_rows),
            "math_book": len(math_rows),
            "strategy_playbooks": len(strat_rows),
            "sat_ops": len(sat_rows),
            "dialogs_seed": len(dialog_rows),
        }
    }, indent=2))

if __name__ == "__main__":
    main()
