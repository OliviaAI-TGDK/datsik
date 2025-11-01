# app/scoring.py
"""
Combined module:
- TGDK Quantum Scoring Utilities (scalar_S + simulate_quantum_with_scorer)
- JD4 -> Scoring-only refactor (no quantum runtime deps)
- FastAPI router: /scoring/model/status, /scoring/model/train, /scoring/score,
                  /scoring/score/batch, /scoring/viz/dependencies.png
"""

from __future__ import annotations

import os
import io
import re
import json
import math
import string
import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional persistence
try:
    from joblib import dump, load  # tiny dependency; skip if not available
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

# ML (lightweight)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Viz (headless)
import matplotlib
matplotlib.use("Agg")  # headless servers
import matplotlib.pyplot as plt
import networkx as nx

# FastAPI router
from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger("scoring")
router = APIRouter()

# ============================================================================
# Part 1: TGDK Quantum Scoring Utilities (standalone helpers)
# ============================================================================

def scalar_S(F: float, L: float, M: float, x: float) -> float:
    """Base scalar S formula."""
    return float((-5.0 * F) - (21.0 * L) + (21.0 * M) - 0.9261 - (x ** 3.12))

def simulate_quantum_with_scorer(F: float, L: float, M: float, x: float, size: int = 16) -> dict:
    """
    Uses S to seed & bias a pseudo 'quantum-like' distribution.
    Returns: S, amplitudes (normalized), probabilities, peak_state.
    """
    S = scalar_S(F, L, M, x)

    rng = np.random.default_rng(abs(int(S * 1e6)) % (2**32))
    raw = rng.normal(loc=S, scale=abs(S) * 0.1 + 1, size=size)

    # Normalize -> amplitudes; probs are |amp|^2
    norm = np.linalg.norm(raw) or 1.0
    amplitudes = raw / norm
    probabilities = np.abs(amplitudes) ** 2

    return {
        "S": S,
        "amplitudes": amplitudes.tolist(),
        "probabilities": probabilities.tolist(),
        "peak_state": int(np.argmax(probabilities)),
    }

# ============================================================================
# Part 2: Scoring-only engine (JD4 purposes preserved, no quantum deps)
# ============================================================================

# -------- Feature engineering --------

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freqs = np.array([text.count(c) for c in set(text)], dtype=float)
    p = freqs / freqs.sum()
    return float(-(p * np.log2(p)).sum())

def char_sum(text: str) -> int:
    return sum(ord(c) for c in text)

def token_stats(text: str) -> Tuple[int, float]:
    tokens = re.findall(r"\w+", text or "")
    if not tokens:
        return 0, 0.0
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    return len(tokens), float(avg_len)

def ratios(text: str) -> Tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    n = len(text)
    alnum = sum(ch.isalnum() for ch in text) / n
    digits = sum(ch.isdigit() for ch in text) / n
    spaces = sum(ch.isspace() for ch in text) / n
    return float(alnum), float(digits), float(spaces)

def printable_ratio(text: str) -> float:
    if not text:
        return 1.0
    printable = set(string.printable)
    return sum(c in printable for c in text) / len(text)

def extract_features(text: str, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Simple, robust feature set:
    - length / tokens / average token length
    - character entropy
    - alnum/digit/space ratios
    - checksum-y features to mimic "mapping"
    - printable ratio
    - optional meta hints (length of 'name', presence of ids)
    """
    text = text or ""
    L = len(text)
    tok_count, tok_avg = token_stats(text)
    entropy = shannon_entropy(text)
    alnum, digits, spaces = ratios(text)
    checksum = char_sum(text) % 1_000_003
    printable = printable_ratio(text)
    # stable hashed features
    h = int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)
    h2 = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    h_norm = (h % 10_000_019) / 10_000_019
    h2_norm = (h2 % 1_000_003) / 1_000_003
    # meta-derived signals (optional)
    mlen = float(len(meta.get("name", ""))) if meta else 0.0
    has_id = 1.0 if (meta and any(k in meta for k in ("id", "sat_id", "task_id"))) else 0.0

    feats = np.array([
        L, tok_count, tok_avg, entropy,
        alnum, digits, spaces, printable,
        checksum / 1_000_003.0, h_norm, h2_norm,
        mlen, has_id
    ], dtype=float)
    return feats

# -------- Heuristic baseline --------

def heuristic_score(feats: np.ndarray) -> float:
    """
    Stable [0,1] score with interpretable biases:
     - Prefer medium length, moderate entropy, reasonable digit ratio, printable text.
    """
    L, tok_count, tok_avg, H, A, D, S, P, C, HN, H2N, MLEN, HASID = feats

    length_term = math.exp(-((L - 150.0) ** 2) / (2 * 200.0 ** 2))
    entropy_term = max(0.0, min(1.0, 1.0 - abs(H - 3.5) / 3.5))
    digit_term = max(0.0, min(1.0, 1.0 - abs(D - 0.1) / 0.1))
    printable_term = P
    token_avg_term = max(0.0, min(1.0, 1.0 - abs(tok_avg - 6.0) / 6.0))
    id_bonus = 0.1 if HASID > 0 else 0.0

    raw = (0.30 * length_term + 0.20 * entropy_term + 0.15 * digit_term +
           0.20 * printable_term + 0.15 * token_avg_term + id_bonus)
    return max(0.0, min(1.0, float(raw)))

# -------- Engine --------

@dataclass
class ScoreResult:
    score: float
    label: str
    explanation: Dict[str, Any]

class ScoringEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("SCORING_MODEL_PATH", "/app/data/scoring_model.joblib")
        self.pipeline: Optional[Pipeline] = None
        self.label_map: Dict[int, str] = {}
        self._load_if_available()

    def _load_if_available(self) -> None:
        if HAS_JOBLIB and self.model_path and os.path.exists(self.model_path):
            try:
                payload = load(self.model_path)
                self.pipeline = payload.get("pipeline")
                self.label_map = payload.get("labels", {})
                logger.info("[scoring] loaded model from %s", self.model_path)
            except Exception as e:
                logger.warning("[scoring] failed to load model: %r", e)

    def _save(self) -> None:
        if HAS_JOBLIB and self.model_path and self.pipeline is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            dump({"pipeline": self.pipeline, "labels": self.label_map}, self.model_path)
            logger.info("[scoring] model saved to %s", self.model_path)

    def score(self, text: str, meta: Optional[Dict[str, Any]] = None) -> ScoreResult:
        feats = extract_features(text, meta)
        if self.pipeline is not None:
            try:
                proba = self.pipeline.predict_proba([feats])[0]
                cls = int(np.argmax(proba))
                score = float(np.max(proba))
                label = self.label_map.get(cls, f"class_{cls}")
                expl = {"mode": "model", "proba": {self.label_map.get(i, f'class_{i}'): float(p)
                                                   for i, p in enumerate(proba)}}
                return ScoreResult(score=score, label=label, explanation=expl)
            except Exception as e:
                logger.warning("[scoring] model predict failed, falling back to heuristic: %r", e)

        # Heuristic fallback
        score = heuristic_score(feats)
        label = "high" if score >= 0.66 else ("medium" if score >= 0.33 else "low")
        expl = {"mode": "heuristic", "features": feats.tolist()}
        return ScoreResult(score=score, label=label, explanation=expl)

    def score_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            text = it.get("text") or it.get("data") or it.get("code") or ""
            meta = {k: v for k, v in it.items() if k not in ("text", "data", "code")}
            r = self.score(text, meta)
            out.append({**it, "score": r.score, "label": r.label, "explanation": r.explanation})
        return out

    def train_from_jsonl(self, path: str, text_key: str = "text", label_key: str = "label") -> Dict[str, Any]:
        """
        Train a small RF classifier on local JSONL:
          {"text": "...", "label": "A"}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        X, y = [], []
        label_ids: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = row.get(text_key, "")
                label = str(row.get(label_key, "unknown"))
                feats = extract_features(text)
                if label not in label_ids:
                    label_ids[label] = len(label_ids)
                X.append(feats)
                y.append(label_ids[label])

        if not X:
            raise ValueError("No training rows found")

        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("rf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ])
        pipe.fit(np.array(X), np.array(y))

        self.pipeline = pipe
        self.label_map = {v: k for k, v in label_ids.items()}
        self._save()

        return {"classes": self.label_map, "samples": len(X)}

# Global engine instance
ENGINE = ScoringEngine()

# -------- Tiny dependency visualization (replaces TGDKCARTOGRAPHER) --------

def render_dependency_png(items: List[Dict[str, Any]], title: str = "Task Dependencies") -> bytes:
    """
    Build a tiny DAG by linking items[i] -> items[i+1] if labels escalate (low->medium->high).
    """
    if not items:
        items = [{"task_id": 1, "label": "low"}, {"task_id": 2, "label": "medium"}, {"task_id": 3, "label": "high"}]

    label_rank = {"low": 0, "medium": 1, "high": 2}
    G = nx.DiGraph()
    for it in items:
        node = it.get("task_id") or it.get("id") or hashlib.md5(
            json.dumps(it, sort_keys=True).encode()).hexdigest()[:6]
        G.add_node(node, label=it.get("label", "low"))

    nodes = list(G.nodes())
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if label_rank.get(G.nodes[a]["label"], 0) <= label_rank.get(G.nodes[b]["label"], 0):
            G.add_edge(a, b)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    colors = [
        "#66bb6a" if G.nodes[n]["label"] == "low"
        else "#ffa726" if G.nodes[n]["label"] == "medium"
        else "#ef5350"
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, node_color=colors)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=1.2)
    plt.title(title)
    plt.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    return buf.getvalue()

# -------- FastAPI endpoints --------

@router.get("/model/status")
def model_status():
    return JSONResponse({
        "has_model": bool(ENGINE.pipeline is not None),
        "labels": ENGINE.label_map,
        "path": ENGINE.model_path,
    })

@router.post("/model/train")
def model_train(
    dataset_path: str = Query(..., description="Path to JSONL with {text,label} rows"),
    text_key: str = Query("text"),
    label_key: str = Query("label"),
):
    try:
        result = ENGINE.train_from_jsonl(dataset_path, text_key=text_key, label_key=label_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"train failed: {e!r}")

@router.post("/score")
def score_one(payload: Dict[str, Any] = Body(...)):
    """
    payload: { "text": "...", "task_id": 123, ...meta }
    """
    text = payload.get("text") or payload.get("data") or payload.get("code") or ""
    meta = {k: v for k, v in payload.items() if k not in ("text", "data", "code")}
    r = ENGINE.score(text, meta)
    return {"score": r.score, "label": r.label, "explanation": r.explanation}

@router.post("/score/batch")
def score_batch(payload: List[Dict[str, Any]] = Body(...)):
    items = ENGINE.score_batch(payload)
    return {"items": items, "count": len(items)}

@router.post("/viz/dependencies.png")
def viz_dependencies(payload: List[Dict[str, Any]] = Body(...)):
    png = render_dependency_png(payload, title="Scored Task Dependencies")
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
