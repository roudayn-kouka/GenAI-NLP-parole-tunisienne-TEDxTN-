"""
Interface « Derja Voice Lab » : parole tunisienne -> texte -> traduction, avec le parcours complet de la voix.

Cinq onglets : Nouvelle analyse, Voix existantes (segments du corpus, comparés à la référence), Historique,
Résultats des expériences, Comment ça marche.

Lancement :
    bash scripts/run_app.sh                # dans WSL : vérifie les modèles chargés, http://localhost:7860
    python app/app.py [--share]            # direct
Depuis un notebook (05_Interface_Demo.ipynb) :
    from app.app import launch; launch(share=True)

Les modèles sont ceux des notebooks (workdir/models/asr_best, ast_best, mt_best). S'ils sont absents, l'interface
utilise les modèles de base et le signale (« REPLI ») dans l'étape concernée.
"""
from __future__ import annotations

import argparse
import base64
import html
import inspect
import io
import re
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import history  # noqa: E402
from app.results_view import SECTIONS  # noqa: E402
from app.scoring import compare_to_reference, reference_html  # noqa: E402
from common.config import AppModels, RESULTS_DIR, SEGMENTS_DIR, TARGET_LANGUAGES  # noqa: E402
from common.pipeline import (EXPLAIN, LLM_CORRECTION_WARNING, LLM_TRANSLATION_WARNING, STEP_MODELS, TITLES,  # noqa: E402
                             TunisianSpeechPipeline, describe_model)
from common.text import code_switch_html  # noqa: E402

STEP_KEYS = ["input", "vad", "features", "asr", "codeswitch", "normalize", "llm", "translate", "result"]
ICONS = {"input": "🎙️", "vad": "〰️", "features": "🌈", "asr": "🧠", "codeswitch": "🔀", "normalize": "🧹",
         "llm": "✨", "translate": "🌍", "result": "✅"}
MODES = {"Pipeline": "pipeline", "Directe": "direct", "Comparer": "compare"}
MODE_HELP = ("Pipeline : la voix devient du texte tunisien, puis ce texte est traduit (NLLB). Directe : un seul "
             "modèle va de l'audio à l'anglais. Comparer : les deux, côte à côte.")

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap" rel="stylesheet">
"""

CSS = """
:root {
  --dv-ink:#1c1917; --dv-muted:#78716c; --dv-line:#e7e5e4; --dv-card:#ffffff; --dv-soft:#faf7f2;
  --dv-accent:#c2410c; --dv-accent-soft:#fff1e8; --dv-teal:#0f766e; --dv-teal-soft:#e6f4f2;
  --dv-warn:#b45309; --dv-warn-soft:#fef3c7; --dv-err:#b91c1c; --dv-err-soft:#fee2e2;
}
.dark {
  --dv-ink:#f5f5f4; --dv-muted:#a8a29e; --dv-line:#3a3532; --dv-card:#1f1c1a; --dv-soft:#171412;
  --dv-accent:#fb923c; --dv-accent-soft:#3b2416; --dv-teal:#5eead4; --dv-teal-soft:#123532;
  --dv-warn:#fbbf24; --dv-warn-soft:#3a2c0c; --dv-err:#f87171; --dv-err-soft:#3b1414;
}
.gradio-container { max-width: 1240px !important; font-family: Inter, 'IBM Plex Sans Arabic', system-ui, sans-serif !important; }
[dir=rtl], [dir=auto] { font-family: 'IBM Plex Sans Arabic', Inter, sans-serif; unicode-bidi: plaintext; }
bdi { unicode-bidi: isolate; }

/* --- en-tête --- */
.dv-hero { position:relative; overflow:hidden; border-radius:18px; padding:20px 26px 18px;
  background: radial-gradient(900px 240px at 100% 0%, #fdba74 0%, transparent 55%),
              linear-gradient(135deg, #7c2d12 0%, #c2410c 55%, #ea580c 100%); color:#fff; }
.dv-hero::after { content:"كلامك"; position:absolute; right:26px; top:2px; font-size:72px; line-height:1;
  font-weight:600; opacity:.13; white-space:nowrap; font-family:'IBM Plex Sans Arabic', Inter; pointer-events:none; }
.dv-hero h1 { margin:0; font-size:25px; font-weight:800; letter-spacing:-.02em; color:#fff; }
.dv-hero p { margin:4px 0 0; font-size:14.5px; opacity:.92; max-width:720px; color:#fff; }

/* --- cartes --- */
.dv-card { background:var(--dv-card); border:1px solid var(--dv-line); border-radius:16px; padding:16px 18px; margin-bottom:12px; }
.dv-label { font-size:11.5px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; color:var(--dv-muted); margin-bottom:8px; }
.dv-big { font-size:20px; line-height:1.8; color:var(--dv-ink); }
.dv-empty { color:var(--dv-muted); font-style:italic; font-size:15px; }
.dv-tr-item { margin-top:12px; padding-top:12px; border-top:1px dashed var(--dv-line); }
.dv-tr-name { font-size:12px; font-weight:700; color:var(--dv-teal); margin-bottom:2px; }
.dv-tr-text { font-size:18px; line-height:1.6; color:var(--dv-ink); }
.dv-meta { font-size:12.5px; color:var(--dv-muted); margin-top:10px; }
.dv-saved { font-size:12.5px; color:var(--dv-teal); margin-top:6px; }
.dv-hint { font-size:13px; color:var(--dv-muted); line-height:1.5; margin:6px 0 10px; }
.dv-sub { font-size:12px; font-weight:700; color:var(--dv-muted); margin:12px 0 2px; text-transform:uppercase; letter-spacing:.05em; }
.dv-ref { border-left:4px solid var(--dv-accent); }
.dv-scores { display:flex; flex-wrap:wrap; gap:8px; }
.dv-score { background:var(--dv-soft); border:1px solid var(--dv-line); border-radius:12px; padding:8px 12px; display:flex; flex-direction:column; min-width:120px; }
.dv-score span { font-size:11.5px; color:var(--dv-muted); } .dv-score b { font-size:16px; color:var(--dv-ink); }
.dv-reftext { font-size:15px; color:var(--dv-ink); }

/* --- progression (frise horizontale compacte) --- */
.dv-progress { display:flex; gap:0; overflow-x:auto; padding:2px 0 6px; }
.dv-pn { flex:1 0 58px; text-align:center; position:relative; }
.dv-pn::before { content:""; position:absolute; top:15px; left:-50%; width:100%; height:2px; background:var(--dv-line); }
.dv-pn:first-child::before { display:none; }
.dv-pn.ok::before, .dv-pn.warn::before { background:var(--dv-accent); }
.dv-pd { position:relative; z-index:1; width:32px; height:32px; margin:0 auto 3px; border-radius:50%; display:flex;
  align-items:center; justify-content:center; font-size:15px; background:var(--dv-card); border:2px solid var(--dv-line); }
.dv-pn.ok .dv-pd { border-color:var(--dv-accent); background:var(--dv-accent-soft); }
.dv-pn.warn .dv-pd { border-color:var(--dv-warn); background:var(--dv-warn-soft); }
.dv-pn.error .dv-pd { border-color:var(--dv-err); background:var(--dv-err-soft); }
.dv-pn.run .dv-pd { border-color:var(--dv-accent); animation: dvpulse 1s infinite; }
.dv-pn.skip { opacity:.35; }
.dv-pt { font-size:10.5px; color:var(--dv-muted); line-height:1.2; }
@keyframes dvpulse { 0%{box-shadow:0 0 0 0 rgba(194,65,12,.45)} 100%{box-shadow:0 0 0 10px rgba(194,65,12,0)} }

/* --- parcours (frise verticale, étapes repliées) --- */
.dv-journey { position:relative; margin:4px 0 8px; }
.dv-journey::before { content:""; position:absolute; left:19px; top:8px; bottom:8px; width:2px; background:var(--dv-line); }
.dv-step { position:relative; margin:0 0 10px 0; }
.dv-step > summary { list-style:none; cursor:pointer; display:flex; gap:12px; align-items:flex-start; padding:4px 0; }
.dv-step > summary::-webkit-details-marker { display:none; }
.dv-sdot { flex:0 0 40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:18px; background:var(--dv-card); border:2px solid var(--dv-accent); position:relative; z-index:1; }
.dv-step.warn .dv-sdot { border-color:var(--dv-warn); background:var(--dv-warn-soft); }
.dv-step.error .dv-sdot { border-color:var(--dv-err); background:var(--dv-err-soft); }
.dv-shead { flex:1; min-width:0; background:var(--dv-card); border:1px solid var(--dv-line); border-radius:14px; padding:10px 14px; }
.dv-step[open] .dv-shead { border-color:var(--dv-accent); }
.dv-stitle { font-weight:700; color:var(--dv-ink); font-size:15px; display:flex; justify-content:space-between; gap:8px; }
.dv-stitle .dv-time { font-weight:500; color:var(--dv-muted); font-size:12px; white-space:nowrap; }
.dv-ssum { color:var(--dv-ink); font-size:13.5px; margin-top:2px; }
.dv-smodel { display:inline-block; margin-top:6px; font-size:12px; color:var(--dv-teal); background:var(--dv-teal-soft);
  border-radius:999px; padding:2px 10px; }
.dv-smodel.fallback { color:var(--dv-warn); background:var(--dv-warn-soft); }
.dv-sbody { margin:6px 0 4px 52px; padding:12px 14px; background:var(--dv-soft); border:1px solid var(--dv-line); border-radius:14px; }
.dv-sexp { color:var(--dv-muted); font-size:13.5px; line-height:1.55; margin-bottom:8px; }
.dv-fig { max-width:100%; border-radius:10px; margin-top:8px; background:#fff; }
.dv-chev { color:var(--dv-muted); font-size:12px; }

/* --- détails des étapes (HTML produit par common/pipeline.py) --- */
.kv-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr)); gap:8px; margin:8px 0 12px; }
.kv { background:var(--dv-card); border:1px solid var(--dv-line); border-radius:10px; padding:7px 10px; display:flex; flex-direction:column; }
.kv span { font-size:11px; color:var(--dv-muted); } .kv b { font-size:14px; color:var(--dv-ink); font-weight:600; }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip { font-size:12px; background:var(--dv-accent-soft); color:var(--dv-accent); padding:3px 9px; border-radius:999px; font-weight:500; }
.tok-line, .cs-line { font-size:18px; line-height:2.1; }
.tok { padding:2px 1px; border-radius:5px; }
.tok-hi { background:rgba(15,118,110,.16); } .tok-mid { background:rgba(234,179,8,.28); } .tok-lo { background:rgba(220,38,38,.28); }
.legend { display:flex; gap:10px; font-size:12px; margin:4px 0 8px; flex-wrap:wrap; }
.legend .tok, .legend .cs-tok { padding:2px 8px; font-size:12px; }
.cs-tok { color:var(--c); background:color-mix(in srgb, var(--c) 12%, transparent); border-bottom:2px solid var(--c);
  padding:1px 5px; border-radius:6px; margin:0 1px; }
.dark .cs-tok { color:color-mix(in srgb, var(--c) 55%, white); }
.chunk-h { font-size:12px; font-weight:600; color:var(--dv-muted); margin:10px 0 2px; }
.chunk-text { font-size:16px; margin-top:4px; }
.fr-line { display:flex; flex-wrap:wrap; gap:2px; font-family:'IBM Plex Sans Arabic', monospace; }
.fr { min-width:16px; height:22px; font-size:12px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:4px; background:rgba(194,65,12, calc(var(--p) * .55)); color:var(--dv-ink); }
.fr-blank { background:var(--dv-soft); color:var(--dv-muted); } .fr-more { font-size:11px; color:var(--dv-muted); align-self:center; margin-left:6px; }
.rules { padding-left:20px; } .rules li { margin-bottom:10px; } .rules li b { font-size:13px; }
.diff { font-size:16px; line-height:1.9; margin-top:2px; color:var(--dv-ink); }
.diff del { background:var(--dv-err-soft); color:var(--dv-err); text-decoration:line-through; border-radius:4px; padding:0 3px; }
.diff ins { background:var(--dv-teal-soft); color:var(--dv-teal); text-decoration:none; border-radius:4px; padding:0 3px; }
.final-norm { margin-top:8px; padding:10px 12px; background:var(--dv-card); border-radius:10px; font-size:17px; }
.tr-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }
.tr-card { border:1px solid var(--dv-line); border-radius:12px; padding:10px 12px; background:var(--dv-card); }
.tr-h { font-size:12px; font-weight:700; color:var(--dv-teal); margin-bottom:4px; } .tr-t { font-size:16px; line-height:1.6; }
.agree { margin-top:10px; font-size:13px; color:var(--dv-muted); }
.note, .warn { font-size:13px; border-radius:10px; padding:8px 12px; margin-bottom:8px; }
.note { background:var(--dv-card); color:var(--dv-muted); border:1px solid var(--dv-line); }
.warn { background:var(--dv-warn-soft); color:var(--dv-warn); border:1px solid color-mix(in srgb, var(--dv-warn) 30%, transparent); }
.err { color:var(--dv-err); white-space:pre-wrap; }

/* --- voix existantes --- */
.dv-voice { border:1px solid var(--dv-line); border-radius:14px; padding:10px 12px; background:var(--dv-card); }
.dv-vmeta { font-size:12px; color:var(--dv-muted); margin-bottom:4px; }
.dv-vtun { font-size:16.5px; line-height:1.7; color:var(--dv-ink); }
.dv-ven { font-size:14px; color:var(--dv-muted); margin-top:2px; }
.dv-voice { margin-top:10px; }
/* tableaux (voix, historique) : police de l'interface (pas la monospace par défaut), arabe lisible */
.dv-vtable *, .dv-htable * { font-family:var(--font) !important; }
.dv-vtable .cell-wrap, .dv-htable .cell-wrap { font-size:13.5px; unicode-bidi:plaintext; }

/* --- résultats, comment ça marche --- */
.dv-cap { font-size:13.5px; line-height:1.6; color:var(--dv-ink); background:var(--dv-soft); border-left:3px solid var(--dv-teal);
  border-radius:8px; padding:9px 12px; margin:6px 0 18px; }
.dv-paper { font-size:11px; font-weight:700; color:var(--dv-warn); background:var(--dv-warn-soft); border-radius:6px; padding:1px 6px; }
.dv-flow { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:8px 0 14px; }
.dv-box { background:var(--dv-card); border:1px solid var(--dv-line); border-radius:14px; padding:10px 12px; }
.dv-box b { display:block; color:var(--dv-ink); font-size:14px; } .dv-box span { font-size:12px; color:var(--dv-muted); }
.dv-box.model { border-color:var(--dv-teal); } .dv-box.model b { color:var(--dv-teal); }
.dv-branch { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.dv-branch > div { border:1px dashed var(--dv-line); border-radius:14px; padding:10px; }
.dv-btitle { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--dv-accent); margin-bottom:6px; }
.dv-table { width:100%; border-collapse:collapse; font-size:13.5px; } .dv-table td, .dv-table th { border-bottom:1px solid var(--dv-line); padding:7px 8px; text-align:left; color:var(--dv-ink); vertical-align:top; }
.dv-table th { color:var(--dv-muted); font-weight:600; }
.dv-foot { text-align:center; color:var(--dv-muted); font-size:12.5px; padding:14px 0 4px; }
#dv-run { font-size:16px !important; font-weight:700 !important; min-height:46px; }

@media (max-width: 900px) {
  .dv-hero::after { display:none; } .dv-hero h1 { font-size:21px; }
  .dv-big { font-size:18px; } .dv-sbody { margin-left:0; } .dv-branch { grid-template-columns:1fr; }
  .dv-journey::before { display:none; } .dv-sdot { flex-basis:32px; height:32px; font-size:15px; }
  .dv-stitle { flex-direction:column; gap:0; } .dv-shead { padding:8px 10px; }
}
"""


def make_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.orange, secondary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "IBM Plex Sans Arabic", "system-ui", "sans-serif"],
    ).set(button_primary_background_fill="linear-gradient(135deg,#c2410c,#ea580c)",
          button_primary_background_fill_hover="linear-gradient(135deg,#9a3412,#c2410c)",
          block_border_width="1px", block_shadow="0 1px 2px rgba(0,0,0,.04)")


# Gradio 6 a déplacé theme/css/head de Blocks() vers launch() : on gère les deux versions.
_STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters
_STYLE = {"theme": make_theme(), "css": CSS, "head": HEAD}


# --------------------------------------------------------------------------
# Rendus HTML
# --------------------------------------------------------------------------
def _png(fig) -> str:
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def progress_html(res, running_key=None, use_llm=False) -> str:
    done = {s.key: s for s in (res.steps if res else [])}
    nodes = []
    for k in STEP_KEYS:
        if k == "llm" and not use_llm and k not in done:
            cls, sub = "skip", "—"
        elif k in done:
            s = done[k]
            cls = s.status if s.status in ("warn", "error") else "ok"
            sub = f"{s.seconds:.1f} s" if s.seconds >= 0.1 else f"{s.seconds * 1000:.0f} ms"
        elif k == running_key:
            cls, sub = "run", "en cours"
        else:
            cls, sub = "", ""
        nodes.append(f"<div class='dv-pn {cls}' title='{TITLES[k]}'><div class='dv-pd'>{ICONS[k]}</div>"
                     f"<div class='dv-pt'>{sub or '&nbsp;'}</div></div>")
    return "<div class='dv-progress'>" + "".join(nodes) + "</div>"


def result_html(res, target: str, meta: str = "") -> str:
    if not res or not (res.transcript or res.translations):
        return ("<div class='dv-card'><div class='dv-label'>Résultat</div>"
                "<div class='dv-empty'>Enregistrez ou chargez une voix, puis lancez l'analyse : la transcription et "
                "la traduction apparaîtront ici.</div></div>")
    tr = res.step("translate")
    agree = tr.data.get("accord chrF") if tr is not None else None
    items = "".join(f"<div class='dv-tr-item'><div class='dv-tr-name'>{html.escape(k)}</div>"
                    f"<div class='dv-tr-text' dir='auto'>{html.escape(v)}</div></div>" for k, v in res.translations.items())
    if agree is not None:
        items += f"<div class='dv-meta'>Accord entre les deux traductions (chrF) : <b>{agree:.0f}</b> / 100</div>"
    transcript = code_switch_html(res.transcript) if res.transcript else "<div class='dv-empty'>…</div>"
    return ("<div class='dv-card'><div class='dv-label'>Transcription tunisienne · mots colorés par écriture</div>"
            f"<div class='dv-big'>{transcript}</div>"
            f"<div class='dv-label' style='margin-top:14px'>Traduction · {html.escape(target)}</div>"
            f"{items or '<div class=dv-empty>…</div>'}"
            f"{f'<div class=dv-meta>{meta}</div>' if meta else ''}</div>")


def journey_html(res) -> str:
    """Frise verticale du parcours : une étape = un bloc replié (titre, résumé, modèle) ; ouvert : explication,
    détails et figure."""
    if not res or not res.steps:
        return ""
    blocks = []
    for i, s in enumerate(res.steps, 1):
        if not hasattr(s, "_png_cache"):
            s._png_cache = _png(s.figure)
        fig = f"<img class='dv-fig' alt='figure : {html.escape(s.title)}' src='data:image/png;base64,{s._png_cache}'>" \
            if s._png_cache else ""
        model = getattr(s, "model", "") or STEP_MODELS.get(s.key, "")
        fb = " fallback" if "REPLI" in model else ""
        flag = " ⚠️" if s.status == "warn" else " ❌" if s.status == "error" else ""
        blocks.append(
            f"<details class='dv-step {s.status}'><summary><div class='dv-sdot'>{ICONS.get(s.key, '•')}</div>"
            f"<div class='dv-shead'><div class='dv-stitle'><span>{i}. {html.escape(s.title)}{flag}</span>"
            f"<span class='dv-time'>{s.seconds:.2f} s · <span class='dv-chev'>détails ▾</span></span></div>"
            f"<div class='dv-ssum' dir='auto'>{html.escape(s.summary)}</div>"
            f"<span class='dv-smodel{fb}'>Modèle : {html.escape(model)}</span></div></summary>"
            f"<div class='dv-sbody'><div class='dv-sexp'>{html.escape(s.explanation)}</div>{s.html}{fig}</div></details>")
    return ("<div class='dv-label' style='margin:6px 0 10px'>Le parcours de votre voix · cliquez une étape pour "
            "voir le détail</div><div class='dv-journey'>" + "".join(blocks) + "</div>")


# --------------------------------------------------------------------------
# Données : segments du corpus (onglet « Voix existantes »)
# --------------------------------------------------------------------------
def load_voices() -> pd.DataFrame:
    """Segments du dev et du test dont l'audio découpé existe (aucun téléchargement)."""
    try:
        from common import data
        df = data.load_corpus(download=False)
        df = df[df.split.isin(["test", "dev"])]
        df = df.assign(path=[str(SEGMENTS_DIR / f"{s}.wav") for s in df.seg_id])
        ok = df.path.map(lambda p: Path(p).exists())
        if not ok.any():   # nom des fichiers avec décalage (cut_segments) : on retrouve les chemins
            df = data.cut_segments(df)
        df = df[df.path.map(lambda p: Path(p).exists())]
        return df[["seg_id", "split", "talk", "seg_type", "duration", "transcription", "translation", "path"]].reset_index(drop=True)
    except Exception as e:   # pas de corpus : l'onglet l'indique
        print("Voix existantes indisponibles :", e)
        return pd.DataFrame(columns=["seg_id", "split", "talk", "seg_type", "duration", "transcription", "translation", "path"])


def filter_voices(df, split, talk, types, dmin, dmax):
    d = df[df.split == split] if split in ("test", "dev") else df
    if talk and talk != "Tous":
        d = d[d.talk == talk]
    if types:
        d = d[d.seg_type.isin(types)]
    return d[(d.duration >= float(dmin)) & (d.duration <= float(dmax))]


VOICE_COLS = ["n°", "talk", "durée", "type", "début de la transcription"]


def _clip(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n].rstrip() + "…"


def talk_name(t: str) -> str:
    """« What_I_can_do_in_Tunisia_X_TEDxDjerba » -> « What I can do in Tunisia X » (affichage seulement)."""
    return re.sub(r"[_\s]*TEDx\w*$", "", str(t)).replace("_", " ").strip() or str(t)


def voice_table(d: pd.DataFrame) -> pd.DataFrame:
    """Tableau compact de « Voix existantes » : une ligne par segment."""
    if d.empty:
        return pd.DataFrame(columns=VOICE_COLS)
    # n° court (« 00003 ») : le split est déjà choisi dans les filtres ; seg_id = f"{split}__{n°}"
    return pd.DataFrame({"n°": [s.split("__")[-1] for s in d.seg_id], "talk": [_clip(talk_name(t), 34) for t in d.talk],
                         "durée": [f"{x:.1f} s" for x in d.duration], "type": d.seg_type.values,
                         "début de la transcription": [_clip(t, 60) for t in d.transcription]})


def voice_count_html(d: pd.DataFrame) -> str:
    return (f"<div class='dv-hint'><b>{len(d)}</b> segment(s) · cliquez une ligne pour l'écouter et l'analyser"
            f"{' · ' + f'{d.duration.sum() / 60:.0f} min' if len(d) else ''}</div>")


def voice_info_html(r) -> str:
    return (f"<div class='dv-vmeta'>{html.escape(r.seg_id)} · {html.escape(r.seg_type)} · {r.duration:.1f} s · "
            f"{html.escape(r.talk)}</div><div class='dv-vtun' dir='auto'>{html.escape(r.transcription)}</div>"
            f"<div class='dv-ven' dir='ltr'>Référence : {html.escape(str(r.translation))}</div>")


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def build_demo(pipeline: TunisianSpeechPipeline | None = None, history_dir: str | Path | None = None) -> gr.Blocks:
    pipe = pipeline or TunisianSpeechPipeline(AppModels())
    hdir = Path(history_dir) if history_dir else history.HISTORY_DIR
    voices = load_voices()
    blocks_kw = {"title": "Derja Voice Lab · TEDxTN"}
    if not _STYLE_IN_LAUNCH:
        blocks_kw.update(_STYLE)

    def set_beams(nb):
        pipe.num_beams = int(nb)
        if getattr(pipe, "_asr", None) is not None and hasattr(pipe._asr, "num_beams"):
            pipe._asr.num_beams = int(nb)

    def analyse(aud, tgt, md, eng, llm, nb, source: dict, ref: dict | None = None):
        """Analyse commune aux onglets : affichage progressif, puis enregistrement dans l'historique.
        Rend (progression, résultat, comparaison à la référence, parcours)."""
        set_beams(nb)
        settings = {"cible": tgt, "mode": md, "moteur": eng, "correction LLM": bool(llm), "beam": int(nb)}
        res = None
        yield progress_html(None, "input", llm), result_html(None, tgt), "", ""
        for res in pipe.run_iter(aud, target=tgt, mode=MODES[md], use_llm=llm, mt_engine=eng):
            done = {s.key for s in res.steps}
            nxt = next((k for k in STEP_KEYS if k not in done and (k != "llm" or llm)), None)
            yield progress_html(res, nxt, llm), result_html(res, tgt), "", journey_html(res)
        ref_card, meta = "", {"source": source["label"], "source_kind": source["kind"], "settings": settings}
        if ref is not None and res is not None:
            trs = res.translations if tgt == "Anglais" else {}
            scores = compare_to_reference(res.transcript, trs, ref["transcription"], ref["translation"])
            ref_card = reference_html(scores, res.transcript, trs, ref["transcription"], ref["translation"])
            if tgt != "Anglais":
                ref_card = ref_card.replace("</div></div>", "</div><p class='dv-hint'>Traduction vers une autre langue "
                                            "que l'anglais : pas de référence, seuls WER et CER sont calculés.</p></div>", 1)
            meta.update({"reference": ref, "scores": scores, "reference_html": ref_card})
        saved = ""
        if res is not None and res.steps:
            try:
                rid = history.save(res, meta, hdir)
                saved = f"Enregistrée dans l'historique ({rid}) · {res.total_seconds:.1f} s"
            except OSError as e:
                saved = f"Historique non enregistré : {e}"
        yield progress_html(res, None, llm), result_html(res, tgt, saved), ref_card, journey_html(res)

    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML("""<div class="dv-hero"><h1>Derja Voice Lab</h1>
          <p>Reconnaissance et traduction de la parole tunisienne (derja, avec français et anglais), entraînées sur
          le corpus TEDxTN. Suivez chaque étape que traverse votre voix.</p></div>""")

        with gr.Tabs():
            # ============================================== 1. Nouvelle analyse
            with gr.Tab("🎙️ Nouvelle analyse"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Votre voix")
                        src_kind = gr.State("micro")
                        audio.upload(lambda: "fichier", None, src_kind)
                        audio.stop_recording(lambda: "micro", None, src_kind)
                        target = gr.Radio(list(TARGET_LANGUAGES), value="Anglais", label="Langue cible")
                        mode = gr.Radio(list(MODES), value="Comparer", label="Méthode", info=MODE_HELP)
                        with gr.Accordion("Options avancées", open=False):
                            beams = gr.Slider(1, 5, value=1, step=1, label="Beam search de la reconnaissance",
                                              info="1 = décodage glouton (celui évalué). Plus grand : plus lent.")
                            mt_engine = gr.Radio(["NLLB", "LLM", "NLLB + LLM"], value="NLLB",
                                                 label="Moteur de traduction du pipeline", info=LLM_TRANSLATION_WARNING)
                            use_llm = gr.Checkbox(value=False, label="Post-correction de la transcription par LLM",
                                                  info=LLM_CORRECTION_WARNING)
                        run = gr.Button("Analyser", variant="primary", elem_id="dv-run")
                    with gr.Column(scale=7, min_width=320):
                        n_prog = gr.HTML(progress_html(None))
                        n_res = gr.HTML(result_html(None, "Anglais"))
                n_ref = gr.HTML(visible=False)
                n_journey = gr.HTML()

                def on_new(aud, tgt, md, eng, llm, nb, kind):
                    if aud is None:
                        raise gr.Error("Enregistrez ou chargez une voix d'abord.")
                    label = "micro" if kind == "micro" else "fichier"
                    yield from analyse(aud, tgt, md, eng, llm, nb, {"kind": label, "label": label})

                run.click(on_new, [audio, target, mode, mt_engine, use_llm, beams, src_kind], [n_prog, n_res, n_ref, n_journey])

            # ============================================== 2. Voix existantes
            with gr.Tab("🗂️ Voix existantes"):
                if voices.empty:
                    gr.Markdown("Aucun segment audio trouvé (`workdir/audio_segments`). Exécutez la section 4 du notebook 01.")
                gr.HTML("<p class='dv-hint'>Segments du corpus TEDxTN (test et dev) avec leur référence. « Analyser » "
                        "lance la même analyse que « Nouvelle analyse » et la compare à la référence (démonstration : "
                        "aucun système ne change).</p>")
                talks_of = lambda s: ["Tous"] + sorted(voices[voices.split == s].talk.unique().tolist())   # noqa: E731
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, min_width=320):
                        with gr.Row():
                            v_split = gr.Radio(["test", "dev"], value="test", label="Split", min_width=120)
                            v_types = gr.CheckboxGroup(["TUN", "MIXED", "FOR"], value=["TUN", "MIXED", "FOR"],
                                                       label="Type", min_width=200)
                        v_talk = gr.Dropdown(talks_of("test"), value="Tous", label="Talk")
                        with gr.Row():
                            v_dmin = gr.Slider(0, 30, value=0, step=0.5, label="Durée min. (s)", min_width=140)
                            v_dmax = gr.Slider(0, 30, value=30, step=0.5, label="Durée max. (s)", min_width=140)
                        with gr.Accordion("Langue cible et méthode", open=False):
                            v_target = gr.Radio(list(TARGET_LANGUAGES), value="Anglais", label="Langue cible")
                            v_mode = gr.Radio(list(MODES), value="Comparer", label="Méthode")
                        v_count = gr.HTML(voice_count_html(filter_voices(voices, "test", "Tous", ["TUN", "MIXED", "FOR"], 0, 30)))
                        v_table = gr.Dataframe(voice_table(filter_voices(voices, "test", "Tous", ["TUN", "MIXED", "FOR"], 0, 30)),
                                               interactive=False, wrap=True, max_height=440, elem_classes="dv-vtable",
                                               column_widths=["11%", "27%", "11%", "13%", "38%"], show_search="search")
                        v_seg = gr.State(None)   # seg_id de la ligne choisie
                        with gr.Column(visible=False, elem_classes="dv-voice") as v_sel:
                            v_info = gr.HTML()
                            v_audio = gr.Audio(interactive=False, show_label=False, container=False)
                            v_go = gr.Button("Analyser ce segment ▸", variant="primary")
                    with gr.Column(scale=6, min_width=320):
                        v_prog = gr.HTML()
                        v_res = gr.HTML("<div class='dv-card'><div class='dv-empty'>Cliquez une ligne du tableau, "
                                        "écoutez le segment puis « Analyser ▸ » : le résultat et la comparaison avec la "
                                        "référence s'afficheront ici.</div></div>")
                        v_ref = gr.HTML()
                        v_journey = gr.HTML()
                v_split.change(lambda s: gr.update(choices=talks_of(s), value="Tous"), v_split, v_talk)

                def v_filter(split, talk, types, dmin, dmax):
                    d = filter_voices(voices, split, talk, types, dmin, dmax)
                    # nouveau filtre : la sélection précédente est retirée (lecteur et bouton masqués)
                    return voice_count_html(d), voice_table(d), None, gr.update(visible=False), None

                for c in (v_split, v_talk, v_types, v_dmin, v_dmax):
                    c.change(v_filter, [v_split, v_talk, v_types, v_dmin, v_dmax],
                             [v_count, v_table, v_seg, v_sel, v_audio])

                def v_select(split, evt: gr.SelectData):
                    # row_value plutôt que l'indice : la ligne reste juste si le tableau est trié ou filtré par la recherche
                    seg = f"{split}__{(evt.row_value or [''])[0]}"
                    hit = voices[voices.seg_id == seg]
                    if hit.empty:
                        return None, gr.update(visible=False), "", None
                    r = hit.iloc[0]
                    return seg, gr.update(visible=True), voice_info_html(r), r.path

                v_table.select(v_select, v_split, [v_seg, v_sel, v_info, v_audio])

                def on_voice(seg, tgt, md):
                    hit = voices[voices.seg_id == seg]
                    if hit.empty:
                        raise gr.Error("Choisissez d'abord un segment dans le tableau.")
                    r = hit.iloc[0]
                    ref = {"seg_id": r.seg_id, "transcription": r.transcription, "translation": str(r.translation)}
                    yield from analyse(r.path, tgt, md, "NLLB", False, 1, {"kind": "corpus", "label": f"corpus {r.seg_id}"}, ref)

                v_go.click(on_voice, [v_seg, v_target, v_mode], [v_prog, v_res, v_ref, v_journey])

            # ============================================== 3. Historique
            with gr.Tab("🕘 Historique") as tab_hist:
                gr.HTML("<p class='dv-hint'>Chaque analyse est enregistrée sur cette machine (dossier "
                        f"<code>{html.escape(str(hdir))}</code>, hors du dépôt git ; l'audio n'est pas conservé). "
                        "Cliquez une ligne (ou choisissez-la dans la liste), puis « Rouvrir » : tout le parcours "
                        "s'affiche sans relancer les modèles.</p>")
                h_table = gr.Dataframe(interactive=False, wrap=True, max_height=320, elem_classes="dv-htable")
                with gr.Row():
                    h_pick = gr.Dropdown([], label="Analyse", scale=5)
                    h_open = gr.Button("Rouvrir", variant="primary", scale=1, min_width=110)
                    h_del = gr.Button("Supprimer", variant="stop", scale=1, min_width=110)
                    h_exp = gr.Button("Exporter (JSON)", scale=1, min_width=140)
                h_file = gr.File(label="Export", visible=False)
                h_msg = gr.HTML()
                h_res = gr.HTML()
                h_ref = gr.HTML()
                h_journey = gr.HTML()

                def h_refresh(keep=None):
                    recs = history.list_records(hdir)
                    tab = pd.DataFrame([{"date": r["date"].replace("T", " "), "source": r["source"], "cible": r["target"],
                                         "méthode": r["mode"], "durée (s)": round(r["seconds"], 1),
                                         "transcription": (r["transcript"][:90] + "…") if len(r["transcript"]) > 90 else r["transcript"]}
                                        for r in recs]) if recs else pd.DataFrame(columns=["date", "source", "cible", "méthode", "durée (s)", "transcription"])
                    choices = [(f"{r['date'].replace('T', ' ')} · {r['source']} · {r['transcript'][:40]}", r["id"]) for r in recs]
                    value = keep if keep in [c[1] for c in choices] else (choices[0][1] if choices else None)
                    return tab, gr.update(choices=choices, value=value)

                def h_reopen(rid):
                    if not rid:
                        return "<p class='dv-hint'>Choisissez une analyse.</p>", "", "", ""
                    rec = history.load(rid, hdir)
                    res = history.to_result(rec)
                    s = rec.get("settings", {})
                    meta = (f"Analyse du {rec['date'].replace('T', ' ')} · source : {html.escape(rec.get('source', ''))} · "
                            f"méthode : {html.escape(str(s.get('mode', '')))} · {rec.get('total_seconds', 0):.1f} s")
                    return "", result_html(res, rec.get("target", ""), meta), rec.get("reference_html", ""), journey_html(res)

                def h_delete(rid):
                    ok = bool(rid) and history.delete(rid, hdir)
                    tab, dd = h_refresh()
                    return tab, dd, ("<p class='dv-hint'>Analyse supprimée.</p>" if ok else "<p class='dv-hint'>Rien à supprimer.</p>"), "", "", ""

                def h_export(rid):
                    if not rid:
                        return gr.update(visible=False)
                    return gr.update(value=history.export_path(rid, hdir), visible=True)

                def h_row(evt: gr.SelectData):
                    # une ligne du tableau choisit l'analyse (date + source, robustes au tri du tableau)
                    row = evt.row_value or []
                    for r in history.list_records(hdir):
                        if row[:2] == [r["date"].replace("T", " "), r["source"]]:
                            return r["id"]
                    return gr.update()

                tab_hist.select(h_refresh, None, [h_table, h_pick])
                h_table.select(h_row, None, h_pick)
                h_open.click(h_reopen, h_pick, [h_msg, h_res, h_ref, h_journey])
                h_del.click(h_delete, h_pick, [h_table, h_pick, h_msg, h_res, h_ref, h_journey])
                h_exp.click(h_export, h_pick, h_file)

            # ============================================== 4. Résultats des expériences
            with gr.Tab("📊 Résultats des expériences"):
                gr.HTML("<p class='dv-hint'>Tableaux et figures produits par les notebooks (<code>results/</code>). Sous "
                        "chacun : ce qu'il montre, comment le lire et le constat principal ; les chiffres sont lus dans "
                        "les fichiers de résultats. Les choix ont été faits sur le dev ; le test n'a été évalué qu'une "
                        "fois. Détail complet : <code>RESULTATS.md</code>.</p>")
                for i, (name, fn) in enumerate(SECTIONS):
                    items = fn(RESULTS_DIR)
                    with gr.Accordion(name, open=i == 0):
                        if not items:
                            gr.Markdown("Aucun résultat trouvé pour cette partie.")
                        for it in items:
                            gr.HTML(f"<div class='dv-sub' style='margin-top:4px'>{html.escape(it.title)}</div>")
                            if it.kind == "table":
                                gr.Dataframe(it.payload, interactive=False, wrap=True, max_height=360)
                            else:
                                gr.Image(it.payload, show_label=False, interactive=False, container=False)
                            gr.HTML(f"<div class='dv-cap'>{it.caption}</div>")

            # ============================================== 5. Comment ça marche
            with gr.Tab("ℹ️ Comment ça marche"):
                m = pipe.models
                gr.HTML(how_it_works_html(m))
                with gr.Accordion("Chaque étape en détail", open=False):
                    gr.HTML("".join(f"<p><b>{i}. {ICONS[k]} {TITLES[k]}</b> : {html.escape(EXPLAIN[k])}</p>"
                                    for i, k in enumerate(STEP_KEYS, 1)))
                with gr.Accordion("Limites à garder en tête", open=False):
                    gr.HTML(LIMITS_HTML)
        gr.HTML("<div class='dv-foot'>Projet GenAI / NLP · FST · corpus TEDxTN (Bougares et al., ArabicNLP 2025, "
                "CC BY-NC-ND 4.0)</div>")
    return demo


def how_it_works_html(m: AppModels) -> str:
    def box(title, sub, model=False):
        return f"<div class='dv-box{' model' if model else ''}'><b>{title}</b><span>{sub}</span></div>"
    flow = ("<div class='dv-flow'>" + box("🎙️ Signal", "mono, 16 kHz") + box("〰️ Détection de parole", "découpe aux silences")
            + box("🌈 Spectrogramme log-Mel", "80 bandes, 100 trames/s") + "</div>")
    branch = ("<div class='dv-branch'><div><div class='dv-btitle'>Pipeline</div><div class='dv-flow'>"
              + box("🧠 Reconnaissance", "Whisper-small fine-tuné → texte tunisien", True)
              + box("🔀🧹 Analyse du texte", "code-switching, normalisation (règles)")
              + box("🌍 Traduction du texte", "NLLB-600M + LoRA (C2)", True)
              + "</div></div><div><div class='dv-btitle'>Directe</div><div class='dv-flow'>"
              + box("🌍 Traduction directe", "Whisper-small E2E-2 : audio → anglais", True)
              + box("↪ Autres langues", "anglais → français / arabe standard (NLLB)") + "</div></div></div>")
    rows = [("Reconnaissance (ASR)", m.asr, "asr", "Écoute la voix et écrit le texte tunisien, en arabe et en latin."),
            ("Traduction directe", m.ast_e2e, "translate", "Traduit l'audio vers l'anglais sans texte intermédiaire."),
            ("Traduction du texte", m.mt, "mt", "Traduit la transcription (pipeline) ; pivot vers le français / l'arabe standard.")]
    trs = "".join(f"<tr><td><b>{r}</b><br><span class='dv-hint'>{why}</span></td><td>{html.escape(_safe_describe(p, role))}"
                  f"<br><code>{html.escape(str(p))}</code></td></tr>" for r, p, role, why in rows)
    trs += (f"<tr><td><b>LLM (option)</b><br><span class='dv-hint'>Post-correction ou traduction ; déconseillé "
            f"(résultats du dev).</span></td><td>{html.escape(m.llm)} (instruct, 4 bits, sans fine-tuning)</td></tr>")
    return ("<div class='dv-card'><div class='dv-label'>Le parcours de la voix</div>" + flow + branch +
            "<p class='dv-hint'>En mode « Comparer », les deux branches tournent et l'interface mesure leur accord.</p></div>"
            "<div class='dv-card'><div class='dv-label'>Le rôle de chaque modèle</div><table class='dv-table'>"
            "<tr><th>Rôle</th><th>Modèle chargé</th></tr>" + trs + "</table></div>")


def _safe_describe(path, role):
    try:
        return describe_model(path, role, not Path(str(path)).exists())
    except Exception:
        return str(path)


LIMITS_HTML = """<ul>
<li>Corpus d'environ 25 h issu de talks TEDx : parole préparée, une centaine de locuteurs. La parole spontanée, un
accent peu représenté ou un micro bruyant dégradent nettement les résultats (le WER d'un talk suit surtout le bruit).</li>
<li>Le tunisien n'a pas d'orthographe standard : une transcription « différente » de la référence peut être correcte.</li>
<li>Seule la traduction anglaise a des références dans TEDxTN : le français et l'arabe standard sont produits sans
modèle entraîné sur ces paires (NLLB sans fine-tuning, ou pivot anglais) et ne sont pas évalués automatiquement.</li>
<li>Les options LLM (Qwen2.5-1.5B) dégradent la correction et la traduction sur le dev : elles sont proposées pour la
démonstration GenAI, avec un avertissement.</li>
<li>Scores de « Voix existantes » : un seul segment, donc très variables ; les scores fiables sont dans « Résultats ».</li>
</ul>"""


def launch(share: bool = False, port: int | None = None, pipeline: TunisianSpeechPipeline | None = None, **kw):
    """Point d'entrée commun (script et notebook)."""
    demo = build_demo(pipeline)
    if _STYLE_IN_LAUNCH:
        kw = {**_STYLE, **kw}
    kw.setdefault("allowed_paths", [str(SEGMENTS_DIR), str(RESULTS_DIR)])
    demo.queue().launch(share=share, server_port=port, **kw)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interface Derja Voice Lab")
    ap.add_argument("--share", action="store_true", help="créer un lien public Gradio")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--asr", help="chemin ou id du modèle ASR")
    ap.add_argument("--ast", help="chemin ou id du modèle de traduction directe")
    ap.add_argument("--mt", help="chemin ou id du modèle NLLB")
    ap.add_argument("--llm", help="id du LLM")
    a = ap.parse_args()
    models = AppModels()
    for attr, val in (("asr", a.asr), ("ast_e2e", a.ast), ("mt", a.mt), ("llm", a.llm)):
        if val:
            setattr(models, attr, val)
    launch(share=a.share, port=a.port, pipeline=TunisianSpeechPipeline(models), server_name="0.0.0.0")
