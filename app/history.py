"""Historique des analyses de l'interface : une analyse = un fichier JSON dans le workdir (jamais dans le dépôt).

Chaque enregistrement contient la date, la source (micro, fichier ou segment du corpus), les réglages, la
transcription, les traductions et tout le parcours (étapes avec leur résumé, modèle, détail HTML et figure en PNG
intégrée), pour rouvrir une analyse sans relancer les modèles. L'audio n'est pas conservé.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from common.config import WORKDIR
from common.pipeline import PipelineResult, Step

HISTORY_DIR = Path(os.environ.get("TEDXTN_HISTORY_DIR", WORKDIR / "app_history"))
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


def _figure_png(fig) -> str:
    """Figure matplotlib -> PNG en base64 (vide si pas de figure)."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _path(rec_id: str, root: Path) -> Path:
    if not _ID.match(rec_id):           # l'identifiant vient de l'interface : pas de chemin arbitraire
        raise ValueError(f"identifiant d'analyse invalide : {rec_id!r}")
    return Path(root) / f"{rec_id}.json"


def save(res: PipelineResult, meta: dict, root: Path | str = HISTORY_DIR) -> str:
    """Enregistre une analyse terminée. `meta` : source, réglages, référence éventuelle, scores éventuels."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    rec_id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    rec = {
        "id": rec_id, "date": now.isoformat(timespec="seconds"), **meta,
        "target": res.target, "transcript": res.transcript, "transcript_normalized": res.transcript_normalized,
        "translations": dict(res.translations), "total_seconds": round(res.total_seconds, 3),
        "steps": [{"key": s.key, "title": s.title, "explanation": s.explanation, "model": getattr(s, "model", ""),
                   "status": s.status, "summary": s.summary, "seconds": round(s.seconds, 4), "html": s.html,
                   "figure_png": _figure_png(s.figure), "data": _jsonable(s.data)} for s in res.steps],
    }
    _path(rec_id, root).write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec_id


def _jsonable(d):
    try:
        json.dumps(d)
        return d
    except TypeError:
        return {str(k): str(v) for k, v in (d or {}).items()}


def list_records(root: Path | str = HISTORY_DIR) -> list[dict]:
    """Résumés des analyses, de la plus récente à la plus ancienne."""
    out = []
    for p in Path(root).glob("*.json"):
        if not _ID.match(p.stem):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": r["id"], "date": r["date"], "source": r.get("source", ""), "target": r.get("target", ""),
                    "mode": r.get("settings", {}).get("mode", ""), "transcript": r.get("transcript", ""),
                    "seconds": r.get("total_seconds", 0.0)})
    return sorted(out, key=lambda r: r["date"], reverse=True)


def load(rec_id: str, root: Path | str = HISTORY_DIR) -> dict:
    return json.loads(_path(rec_id, Path(root)).read_text(encoding="utf-8"))


def delete(rec_id: str, root: Path | str = HISTORY_DIR) -> bool:
    p = _path(rec_id, Path(root))
    if p.exists():
        p.unlink()
        return True
    return False


def export_path(rec_id: str, root: Path | str = HISTORY_DIR) -> str:
    """Chemin du fichier JSON de l'analyse (téléchargé tel quel par l'interface)."""
    p = _path(rec_id, Path(root))
    if not p.exists():
        raise FileNotFoundError(rec_id)
    return str(p)


def to_result(rec: dict) -> PipelineResult:
    """Reconstruit le parcours d'une analyse enregistrée (figures intégrées au HTML de l'étape)."""
    res = PipelineResult(target=rec.get("target", "Anglais"), transcript=rec.get("transcript", ""),
                         transcript_normalized=rec.get("transcript_normalized", ""),
                         translations=dict(rec.get("translations", {})), total_seconds=rec.get("total_seconds", 0.0))
    for s in rec.get("steps", []):
        html = s.get("html", "")
        if s.get("figure_png"):
            html += f"<img class='dv-fig' alt='figure de l étape' src='data:image/png;base64,{s['figure_png']}'>"
        st = Step(s["key"], s.get("title", ""), s.get("explanation", ""), seconds=s.get("seconds", 0.0),
                  status=s.get("status", "ok"), summary=s.get("summary", ""), html=html, data=s.get("data", {}))
        if hasattr(st, "model"):
            st.model = s.get("model", "")
        res.steps.append(st)
    return res
