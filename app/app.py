"""
Interface « Derja Voice Lab » : parole tunisienne -> texte -> traduction, avec le parcours
complet de la voix étape par étape.

Lancement :
    python app/app.py                      # http://localhost:7860
    python app/app.py --share              # lien public (Colab)
Depuis un notebook (voir 05_Interface_Demo.ipynb) :
    from app.app import launch; launch(share=True)

Les modèles sont ceux produits par les notebooks (workdir/models/...). S'ils sont absents,
l'interface utilise les modèles de base et le signale dans l'étape concernée.
"""
from __future__ import annotations

import argparse
import html
import inspect
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import AppModels, RESULTS_DIR, SEGMENTS_DIR, TARGET_LANGUAGES  # noqa: E402
from common.pipeline import EXPLAIN, TITLES, TunisianSpeechPipeline  # noqa: E402
from common.text import code_switch_html  # noqa: E402

STEP_KEYS = ["input", "vad", "features", "asr", "codeswitch", "normalize", "llm", "translate", "result"]
ICONS = {"input": "🎙️", "vad": "〰️", "features": "🌈", "asr": "🧠", "codeswitch": "🔀", "normalize": "🧹",
         "llm": "✨", "translate": "🌍", "result": "✅"}
MODES = {"Pipeline (ASR → traduction)": "pipeline", "Directe (end-to-end)": "direct",
         "Comparer les deux": "compare"}

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
.gradio-container { max-width: 1280px !important; font-family: Inter, 'IBM Plex Sans Arabic', system-ui, sans-serif !important; }
[dir=rtl], [dir=auto] { font-family: 'IBM Plex Sans Arabic', Inter, sans-serif; }

/* --- en-tête --- */
.dv-hero { position:relative; overflow:hidden; border-radius:22px; padding:30px 32px 26px;
  background: radial-gradient(1200px 300px at 100% 0%, #fdba74 0%, transparent 55%),
              linear-gradient(135deg, #7c2d12 0%, #c2410c 55%, #ea580c 100%); color:#fff; }
.dv-hero::after { content:"كلامك"; position:absolute; right:34px; top:6px; font-size:96px; line-height:1;
  font-weight:600; opacity:.13; white-space:nowrap; font-family:'IBM Plex Sans Arabic', Inter; pointer-events:none; }
.dv-hero h1 { margin:0; font-size:30px; font-weight:800; letter-spacing:-.02em; color:#fff; }
.dv-hero p { margin:6px 0 0; font-size:15px; opacity:.92; max-width:760px; color:#fff; }
.dv-badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.dv-badge { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); padding:4px 11px;
  border-radius:999px; font-size:12.5px; font-weight:500; color:#fff; }

/* --- cartes résultat --- */
.dv-card { background:var(--dv-card); border:1px solid var(--dv-line); border-radius:16px; padding:16px 18px; }
.dv-label { font-size:11.5px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; color:var(--dv-muted); margin-bottom:8px; }
.dv-big { font-size:21px; line-height:1.75; color:var(--dv-ink); }
.dv-empty { color:var(--dv-muted); font-style:italic; }
.dv-tr { border-left:4px solid var(--dv-teal); background:var(--dv-teal-soft); }
.dv-tr .dv-big { font-size:19px; }
.dv-tr-item + .dv-tr-item { margin-top:12px; padding-top:12px; border-top:1px dashed var(--dv-line); }
.dv-tr-name { font-size:12px; font-weight:600; color:var(--dv-teal); margin-bottom:2px; }

/* --- timeline --- */
.dv-timeline { display:flex; gap:0; overflow-x:auto; padding:6px 2px 10px; }
.dv-node { flex:1 0 76px; position:relative; text-align:center; padding:0 4px; }
.dv-node::before { content:""; position:absolute; top:21px; left:-50%; width:100%; height:3px; background:var(--dv-line); z-index:0; }
.dv-node:first-child::before { display:none; }
.dv-node.ok::before, .dv-node.warn::before { background:var(--dv-accent); }
.dv-dot { position:relative; z-index:1; width:44px; height:44px; margin:0 auto 6px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:19px; background:var(--dv-card);
  border:2px solid var(--dv-line); transition:all .3s; }
.dv-node.ok .dv-dot { border-color:var(--dv-accent); background:var(--dv-accent-soft); }
.dv-node.warn .dv-dot { border-color:var(--dv-warn); background:var(--dv-warn-soft); }
.dv-node.error .dv-dot { border-color:var(--dv-err); background:var(--dv-err-soft); }
.dv-node.run .dv-dot { border-color:var(--dv-accent); animation: dvpulse 1s infinite; }
.dv-node.skip { opacity:.38; }
@keyframes dvpulse { 0%{box-shadow:0 0 0 0 rgba(194,65,12,.45)} 100%{box-shadow:0 0 0 12px rgba(194,65,12,0)} }
.dv-nt { font-size:11.5px; font-weight:600; color:var(--dv-ink); line-height:1.25; }
.dv-ns { font-size:11px; color:var(--dv-muted); margin-top:2px; }

/* --- détail des étapes --- */
.dv-step-head { display:flex; gap:12px; align-items:flex-start; margin-bottom:10px; }
.dv-step-num { flex:0 0 30px; height:30px; border-radius:9px; background:var(--dv-accent); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-size:14px; }
.dv-step-exp { color:var(--dv-muted); font-size:14px; line-height:1.55; }
.dv-step-sum { font-weight:600; color:var(--dv-ink); margin-bottom:3px; }
.kv-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:8px; margin:8px 0 12px; }
.kv { background:var(--dv-soft); border:1px solid var(--dv-line); border-radius:10px; padding:7px 10px; display:flex; flex-direction:column; }
.kv span { font-size:11px; color:var(--dv-muted); }
.kv b { font-size:14px; color:var(--dv-ink); font-weight:600; }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip { font-size:12px; background:var(--dv-accent-soft); color:var(--dv-accent); padding:3px 9px; border-radius:999px; font-weight:500; }
.tok-line, .cs-line { font-size:18px; line-height:2.1; }
.tok { padding:2px 1px; border-radius:5px; }
.tok-hi { background:rgba(15,118,110,.14); } .tok-mid { background:rgba(234,179,8,.25); } .tok-lo { background:rgba(220,38,38,.25); }
.legend { display:flex; gap:10px; font-size:12px; margin:4px 0 8px; flex-wrap:wrap; }
.legend .tok, .legend .cs-tok { padding:2px 8px; font-size:12px; }
.cs-tok { color:var(--c); background:color-mix(in srgb, var(--c) 12%, transparent); border-bottom:2px solid var(--c);
  padding:1px 5px; border-radius:6px; margin:0 1px; }
.chunk-h { font-size:12px; font-weight:600; color:var(--dv-muted); margin:10px 0 2px; }
.chunk-text { font-size:16px; margin-top:4px; }
.fr-line { display:flex; flex-wrap:wrap; gap:2px; font-family:'IBM Plex Sans Arabic', monospace; }
.fr { min-width:16px; height:22px; font-size:12px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:4px; background:rgba(194,65,12, calc(var(--p) * .55)); color:var(--dv-ink); }
.fr-blank { background:var(--dv-soft); color:var(--dv-muted); }
.fr-more { font-size:11px; color:var(--dv-muted); align-self:center; margin-left:6px; }
.rules { padding-left:20px; } .rules li { margin-bottom:10px; } .rules li b { font-size:13px; }
.diff { font-size:16px; line-height:1.9; margin-top:2px; }
.diff del { background:var(--dv-err-soft); color:var(--dv-err); text-decoration:line-through; border-radius:4px; padding:0 3px; }
.diff ins { background:var(--dv-teal-soft); color:var(--dv-teal); text-decoration:none; border-radius:4px; padding:0 3px; }
.final-norm { margin-top:8px; padding:10px 12px; background:var(--dv-soft); border-radius:10px; font-size:17px; }
.tr-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }
.tr-card { border:1px solid var(--dv-line); border-radius:12px; padding:10px 12px; background:var(--dv-card); }
.tr-h { font-size:12px; font-weight:700; color:var(--dv-teal); margin-bottom:4px; }
.tr-t { font-size:16px; line-height:1.6; }
.agree { margin-top:10px; font-size:13px; color:var(--dv-muted); }
.note, .warn { font-size:13px; border-radius:10px; padding:8px 12px; margin-bottom:8px; }
.note { background:var(--dv-soft); color:var(--dv-muted); border:1px solid var(--dv-line); }
.warn { background:var(--dv-warn-soft); color:var(--dv-warn); border:1px solid color-mix(in srgb, var(--dv-warn) 30%, transparent); }
.err { color:var(--dv-err); white-space:pre-wrap; }
.dv-foot { text-align:center; color:var(--dv-muted); font-size:12.5px; padding:14px 0 4px; }
#dv-run { font-size:16px !important; font-weight:700 !important; min-height:48px; }
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
# Rendus
# --------------------------------------------------------------------------
def timeline_html(res, running_key=None, use_llm=False) -> str:
    done = {s.key: s for s in (res.steps if res else [])}
    nodes = []
    for k in STEP_KEYS:
        if k == "llm" and not use_llm:
            cls, sub = "skip", "désactivé"
        elif k in done:
            s = done[k]
            cls = s.status if s.status in ("warn", "error") else "ok"
            sub = f"{s.seconds * 1000:,.0f} ms".replace(",", " ")
        elif k == running_key:
            cls, sub = "run", "en cours…"
        else:
            cls, sub = "", "en attente"
        nodes.append(f"<div class='dv-node {cls}'><div class='dv-dot'>{ICONS[k]}</div>"
                     f"<div class='dv-nt'>{TITLES[k]}</div><div class='dv-ns'>{sub}</div></div>")
    return "<div class='dv-timeline'>" + "".join(nodes) + "</div>"


def transcript_card(res) -> str:
    if not res or not res.transcript:
        return "<div class='dv-card'><div class='dv-label'>Transcription tunisienne</div><div class='dv-big dv-empty'>La transcription apparaîtra ici.</div></div>"
    return ("<div class='dv-card'><div class='dv-label'>Transcription tunisienne · mots colorés par script</div>"
            f"<div class='dv-big'>{code_switch_html(res.transcript)}</div></div>")


def translation_card(res, target) -> str:
    if not res or not res.translations:
        return (f"<div class='dv-card dv-tr'><div class='dv-label'>Traduction · {html.escape(target)}</div>"
                "<div class='dv-big dv-empty'>La traduction apparaîtra ici.</div></div>")
    items = "".join(f"<div class='dv-tr-item'><div class='dv-tr-name'>{html.escape(k)}</div>"
                    f"<div class='dv-big' dir='auto'>{html.escape(v)}</div></div>" for k, v in res.translations.items())
    return f"<div class='dv-card dv-tr'><div class='dv-label'>Traduction · {html.escape(target)}</div>{items}</div>"


def step_html(i, step) -> str:
    return (f"<div class='dv-step-head'><div class='dv-step-num'>{i}</div><div>"
            f"<div class='dv-step-sum'>{html.escape(step.summary)}</div>"
            f"<div class='dv-step-exp'>{html.escape(step.explanation)}</div></div></div>{step.html}")


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def _examples():
    files = sorted(SEGMENTS_DIR.glob("test__*.wav"))[:6]
    return [[str(f)] for f in files]


def _results_tables():
    out = {}
    for name in ["asr_results", "ast_results", "normalization_results", "dialect_id_results"]:
        p = RESULTS_DIR / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    return out


def build_demo(pipeline: TunisianSpeechPipeline | None = None) -> gr.Blocks:
    pipe = pipeline or TunisianSpeechPipeline(AppModels())
    blocks_kw = {"title": "Derja Voice Lab · TEDxTN"}
    if not _STYLE_IN_LAUNCH:
        blocks_kw.update(_STYLE)

    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML("""
        <div class="dv-hero">
          <h1>Derja Voice Lab</h1>
          <p>Reconnaissance et traduction de la parole tunisienne, entraînées sur le corpus TEDxTN.
             Parlez en derja (même en mélangeant français et anglais) et suivez chaque étape que traverse votre voix.</p>
          <div class="dv-badges"><span class="dv-badge">ASR fine-tuné</span><span class="dv-badge">Traduction directe ou en pipeline</span>
          <span class="dv-badge">Anglais · Français · Arabe standard</span><span class="dv-badge">Code-switching analysé</span></div>
        </div>""")

        with gr.Tabs():
            # ------------------------------------------------ Analyse
            with gr.Tab("🎙️ Analyser une voix"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320):
                        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Votre voix (tunisien)")
                        target = gr.Dropdown(list(TARGET_LANGUAGES), value="Anglais", label="Traduire vers")
                        mode = gr.Radio(list(MODES), value="Comparer les deux", label="Méthode de traduction")
                        with gr.Accordion("Options avancées", open=False):
                            mt_engine = gr.Radio(["NLLB", "LLM", "NLLB + LLM"], value="NLLB",
                                                 label="Moteur de traduction du pipeline")
                            use_llm = gr.Checkbox(value=False, label="Post-correction de la transcription par LLM (GenAI)")
                            beams = gr.Slider(1, 5, value=1, step=1, label="Beam search (Whisper)")
                        run = gr.Button("Analyser ma voix", variant="primary", elem_id="dv-run")
                        ex = _examples()
                        if ex:
                            gr.Examples(ex, inputs=[audio], label="Segments du test TEDxTN")
                    with gr.Column(scale=7):
                        tl = gr.HTML(timeline_html(None))
                        tr_card = gr.HTML(transcript_card(None))
                        tl_card = gr.HTML(translation_card(None, "Anglais"))

                gr.Markdown("### Le parcours de votre voix, étape par étape")
                accs, htmls, plots = [], [], []
                for i, k in enumerate(STEP_KEYS, 1):
                    with gr.Accordion(f"{i}. {ICONS[k]} {TITLES[k]}", open=k in ("asr", "translate"), visible=False) as acc:
                        h = gr.HTML()
                        p = gr.Plot(visible=False, show_label=False)
                    accs.append(acc)
                    htmls.append(h)
                    plots.append(p)

                def on_run(aud, tgt, md, eng, llm, nb):
                    if aud is None:
                        raise gr.Error("Enregistrez ou chargez un audio d'abord.")
                    pipe.num_beams = int(nb)
                    if getattr(pipe, "_asr", None) is not None and hasattr(pipe._asr, "num_beams"):
                        pipe._asr.num_beams = int(nb)
                    hidden = [gr.update(visible=False)] * (3 * len(STEP_KEYS))
                    yield [timeline_html(None, "input", llm), transcript_card(None), translation_card(None, tgt), *hidden]
                    for res in pipe.run_iter(aud, target=tgt, mode=MODES[md], use_llm=llm, mt_engine=eng):
                        done = {s.key for s in res.steps}
                        nxt = next((k for k in STEP_KEYS if k not in done and (k != "llm" or llm)), None)
                        ups_acc, ups_h, ups_p = [], [], []
                        for i, k in enumerate(STEP_KEYS, 1):
                            s = res.step(k)
                            if s is None:
                                ups_acc.append(gr.update(visible=False))
                                ups_h.append(gr.update())
                                ups_p.append(gr.update(visible=False))
                                continue
                            flag = " ⚠️" if s.status == "warn" else " ❌" if s.status == "error" else ""
                            ups_acc.append(gr.update(visible=True, label=f"{i}. {ICONS[k]} {s.title}{flag} · {s.summary}"))
                            ups_h.append(gr.update(value=step_html(i, s), visible=True))
                            ups_p.append(gr.update(value=s.figure, visible=s.figure is not None))
                        yield [timeline_html(res, nxt, llm), transcript_card(res), translation_card(res, tgt),
                               *ups_acc, *ups_h, *ups_p]

                run.click(on_run, [audio, target, mode, mt_engine, use_llm, beams],
                          [tl, tr_card, tl_card, *accs, *htmls, *plots])

            # ------------------------------------------------ Résultats
            with gr.Tab("📊 Résultats des expériences"):
                tables = _results_tables()
                if not tables:
                    gr.Markdown("Aucun résultat trouvé dans `results/`. Exécutez les notebooks 01 à 04 : chacun "
                                "enregistre ses tableaux (`asr_results.csv`, `ast_results.csv`, ...) que cet onglet affiche.")
                labels = {"asr_results": "Reconnaissance (WER / CER, test)", "ast_results": "Traduction (BLEU / chrF++ / COMET, test)",
                          "normalization_results": "Normalisation", "dialect_id_results": "Identification dialectale"}
                for k, df in tables.items():
                    gr.Markdown(f"#### {labels.get(k, k)}")
                    gr.Dataframe(df, interactive=False, wrap=True)
                for fig in sorted(RESULTS_DIR.glob("*.png"))[:8]:
                    gr.Image(str(fig), label=fig.stem, show_label=True)

            # ------------------------------------------------ À propos
            with gr.Tab("ℹ️ Comment ça marche"):
                steps_md = "\n".join(f"{i}. **{TITLES[k]}** : {EXPLAIN[k]}" for i, k in enumerate(STEP_KEYS, 1))
                m = pipe.models
                gr.Markdown(f"""
### Les étapes
{steps_md}

### Modèles utilisés
| Rôle | Modèle (repli si absent) |
|---|---|
| ASR | `{m.asr}` (`{m.asr_fallback}`) |
| Traduction directe | `{m.ast_e2e}` (`{m.ast_e2e_fallback}`) |
| Traduction texte | `{m.mt}` (`{m.mt_fallback}`) |
| LLM (option) | `{m.llm}` |

### Limites à garder en tête
- Corpus d'environ 25 h issu de talks TEDx : parole préparée, 11 villes, 130 locuteurs. La parole spontanée,
  les accents peu représentés ou un micro bruyant dégradent les résultats.
- Le tunisien n'a pas d'orthographe standard : une transcription « différente » de la référence peut être correcte.
- Seule la traduction anglaise a des références dans TEDxTN : le français et l'arabe standard sont produits sans
  modèle entraîné sur ces paires (NLLB/LLM en zero-shot, ou pivot anglais) et n'ont pas pu être évalués automatiquement.
""")
        gr.HTML("<div class='dv-foot'>Projet GenAI / NLP · FST · corpus TEDxTN (Bougares et al., ArabicNLP 2025, CC BY-NC-ND 4.0)</div>")
    return demo


def launch(share: bool = False, port: int | None = None, pipeline: TunisianSpeechPipeline | None = None, **kw):
    """Point d'entrée commun (script et notebook)."""
    demo = build_demo(pipeline)
    if _STYLE_IN_LAUNCH:
        kw = {**_STYLE, **kw}
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
