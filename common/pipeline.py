"""
Pipeline complet « parole tunisienne -> texte -> traduction » avec trace étape par étape.

Chaque étape produit un objet `Step` (titre, explication, durée, sorties texte/HTML/figure)
afin que l'interface puisse montrer par où passe la voix avant d'arriver au résultat :

  1. Signal d'entrée        : conversion mono 16 kHz, mesure du niveau, normalisation
  2. Détection de parole     : VAD par énergie, découpage en morceaux <= 25 s (fenêtre Whisper)
  3. Représentation acoustique : spectrogramme log-Mel (ce que « voit » le modèle)
  4. Reconnaissance (ASR)    : modèle fine-tuné, texte + confiance par token / par trame
  5. Code-switching          : script de chaque mot, type de segment, CMI
  6. Normalisation           : règles appliquées une par une
  7. Post-correction LLM     : (optionnel) correction GenAI
  8. Traduction              : directe (end-to-end), pipeline (NLLB / LLM) ou comparaison
  9. Résultat                : texte final + temps par étape

Les modèles sont encapsulés dans des « backends » remplaçables (les tests utilisent des faux backends).
"""
from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import AppModels, LM_DIR, SAMPLING_RATE, TARGET_LANGUAGES
from .data import load_audio
from .text import WRITING_NORM, cmi, code_switch_html, code_switch_stats, normalize_text, segment_type

MAX_CHUNK_S = 25.0

# Avertissements des options LLM (mesures du DEV de TEDxTN, RESULTATS.md ; affichés dans l'interface et dans l'étape)
LLM_CORRECTION_WARNING = (
    "Option déconseillée : sur le dev de TEDxTN, la post-correction par LLM (Qwen2.5-1.5B) dégrade la transcription "
    "(WER 34,8 % → 42,5 %, +7,7 points ; IC 95 % apparié [+6,7 ; +8,7]). Elle n'a pas été retenue.")
LLM_TRANSLATION_WARNING = (
    "Option déconseillée : sur le dev de TEDxTN, la traduction par LLM (Qwen2.5-1.5B) obtient un BLEU de 6,1 contre "
    "21,1 pour NLLB fine-tuné (chrF++ 27,5 contre 45,3).")


# ==========================================================================
# Structures de résultat
# ==========================================================================
@dataclass
class Step:
    key: str
    title: str
    explanation: str
    seconds: float = 0.0
    status: str = "ok"            # ok | warn | skip | error
    summary: str = ""             # une ligne affichée dans la timeline
    html: str = ""                # détail riche
    figure: Any = None            # figure matplotlib éventuelle
    data: dict = field(default_factory=dict)
    model: str = ""               # modèle utilisé à cette étape (nom et type), affiché par l'interface


@dataclass
class PipelineResult:
    steps: list[Step] = field(default_factory=list)
    transcript: str = ""
    transcript_normalized: str = ""
    translations: dict = field(default_factory=dict)   # {nom_methode: texte}
    target: str = "Anglais"
    total_seconds: float = 0.0

    def step(self, key):
        return next((s for s in self.steps if s.key == key), None)


# ==========================================================================
# Traitement du signal (sans dépendance lourde)
# ==========================================================================
def frame_energy_db(wav: np.ndarray, sr: int, frame_s: float = 0.03) -> np.ndarray:
    n = max(1, int(frame_s * sr))
    k = len(wav) // n
    if k == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-10)])
    fr = wav[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-10)


def energy_vad(wav: np.ndarray, sr: int = SAMPLING_RATE, frame_s: float = 0.03, min_gap_s: float = 0.35,
               min_speech_s: float = 0.2, max_chunk_s: float = MAX_CHUNK_S, pad_s: float = 0.15):
    """Détection d'activité vocale par énergie + découpage en morceaux <= max_chunk_s.
    Renvoie (régions de parole [(début, fin)], morceaux [(début, fin)], énergie par trame, seuil)."""
    e = frame_energy_db(wav, sr, frame_s)
    thr = max(np.percentile(e, 15) + 8.0, e.max() - 45.0)
    speech = e > thr
    regions, start = [], None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        elif not s and start is not None:
            regions.append([start * frame_s, i * frame_s])
            start = None
    if start is not None:
        regions.append([start * frame_s, len(speech) * frame_s])
    merged = []
    for r in regions:  # combler les petites pauses
        if merged and r[0] - merged[-1][1] < min_gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] >= min_speech_s]
    dur = len(wav) / sr
    if not merged:
        merged = [[0.0, dur]]
    chunks = []
    for a, b in merged:  # regrouper les régions en morceaux de taille compatible avec le modèle
        a, b = max(0.0, a - pad_s), min(dur, b + pad_s)
        while b - a > max_chunk_s:
            chunks.append((a, a + max_chunk_s))
            a += max_chunk_s
        if chunks and b - chunks[-1][0] <= max_chunk_s and a - chunks[-1][1] < 1.0:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    return [tuple(r) for r in merged], chunks, e, thr


def mel_filterbank(sr: int = SAMPLING_RATE, n_fft: int = 400, n_mels: int = 80) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def log_mel(wav: np.ndarray, sr: int = SAMPLING_RATE, n_mels: int = 80) -> np.ndarray:
    """Spectrogramme log-Mel avec les paramètres de Whisper (fenêtre 25 ms, pas 10 ms, 80 bandes)."""
    n_fft, hop = 400, 160
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(wav) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(wav[idx] * win, axis=1)) ** 2
    mel = mel_filterbank(sr, n_fft, n_mels) @ spec.T
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return (logm + 4.0) / 4.0


# ==========================================================================
# Figures
# ==========================================================================
PALETTE = {"ink": "#1e293b", "muted": "#64748b", "accent": "#c2410c", "accent2": "#0f766e",
           "grid": "#e2e8f0", "fill": "#fde7d9"}


def _fig(w=9, h=2.4):
    # Figure créée hors de pyplot : pas d'accumulation en mémoire dans une interface qui tourne longtemps
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=110)
    ax = fig.subplots()
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    return fig, ax


def waveform_figure(wav, sr, regions=None, chunks=None, title=""):
    fig, ax = _fig()
    t = np.arange(len(wav)) / sr
    step = max(1, len(wav) // 6000)
    ax.plot(t[::step], wav[::step], color=PALETTE["ink"], lw=0.6)
    for a, b in regions or []:
        ax.axvspan(a, b, color=PALETTE["accent2"], alpha=0.12, lw=0)
    for i, (a, b) in enumerate(chunks or []):
        ax.axvline(a, color=PALETTE["accent"], lw=1, ls="--")
        ax.text(a, 0.97, f" #{i + 1}", color=PALETTE["accent"], fontsize=8,
                transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlim(0, max(t[-1] if len(t) else 1, 1e-3))
    ax.set_xlabel("temps (s)", color=PALETTE["muted"], fontsize=8)
    ax.set_yticks([])
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=PALETTE["ink"])
    fig.tight_layout()
    return fig


def spectrogram_figure(mel, title="Spectrogramme log-Mel (80 bandes, 10 ms/trame)"):
    fig, ax = _fig(9, 2.6)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma",
              extent=[0, mel.shape[1] / 100, 0, mel.shape[0]])
    ax.set_xlabel("temps (s)", color=PALETTE["muted"], fontsize=8)
    ax.set_ylabel("bande Mel", color=PALETTE["muted"], fontsize=8)
    ax.set_title(title, loc="left", fontsize=9, color=PALETTE["ink"])
    fig.tight_layout()
    return fig


def timing_figure(steps: list[Step]):
    shown = [s for s in steps if s.status != "skip" and s.key != "result"]
    fig, ax = _fig(9, 0.38 * len(shown) + 0.6)
    names = [s.title for s in shown][::-1]
    vals = [s.seconds * 1000 for s in shown][::-1]
    ax.barh(names, vals, color=PALETTE["accent"], height=0.55)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:,.0f} ms", va="center", fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("durée (ms)", color=PALETTE["muted"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8, colors=PALETTE["ink"])
    fig.tight_layout()
    return fig


# ==========================================================================
# Rendus HTML
# ==========================================================================
def confidence_html(tokens: list[str], probs: list[float]) -> str:
    spans = []
    for t, p in zip(tokens, probs):
        level = "hi" if p >= 0.8 else "mid" if p >= 0.5 else "lo"
        spans.append(f'<span class="tok tok-{level}" title="p = {p:.2f}">{html.escape(t)}</span>')
    return '<div class="tok-line" dir="auto">' + "".join(spans) + "</div>"


def ctc_frames_html(frames: list[tuple[str, float]], pad: str = "[PAD]", max_frames: int = 400) -> str:
    cells = []
    for c, p in frames[:max_frames]:
        if c == pad:
            cells.append('<span class="fr fr-blank" title="blank">·</span>')
        else:
            ch = " " if c == "|" else c
            cells.append(f'<span class="fr" style="--p:{p:.2f}" title="{html.escape(c)} p={p:.2f}">{html.escape(ch) or "␣"}</span>')
    more = f'<span class="fr-more">… +{len(frames) - max_frames} trames</span>' if len(frames) > max_frames else ""
    return '<div class="fr-line" dir="ltr">' + "".join(cells) + more + "</div>"


def _ltr_label(label: str) -> str:
    """Isole chaque mot arabe d'une étiquette mixte (« alef (أ إ → ا) ») pour garder l'ordre de lecture."""
    import re
    return re.sub(r"([\u0600-\u06FF]+)", r"<bdi>\1</bdi>", html.escape(label))


def kv_html(d: dict) -> str:
    rows = "".join(f"<div class='kv'><span>{html.escape(str(k))}</span><b>{html.escape(str(v))}</b></div>"
                   for k, v in d.items())
    return f"<div class='kv-grid'>{rows}</div>"


def diff_html(a: str, b: str) -> str:
    from difflib import SequenceMatcher
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, aw, bw).get_opcodes():
        if op == "equal":
            out.append(html.escape(" ".join(aw[i1:i2])))
        if op in ("replace", "delete"):
            out.append(f"<del>{html.escape(' '.join(aw[i1:i2]))}</del>")
        if op in ("replace", "insert"):
            out.append(f"<ins>{html.escape(' '.join(bw[j1:j2]))}</ins>")
    return '<div class="diff" dir="auto">' + " ".join(out) + "</div>"


# ==========================================================================
# Description des modèles (affichée dans l'interface ; ne change aucun calcul)
# ==========================================================================
STEP_MODELS = {   # étapes sans modèle appris
    "input": "Traitement du signal, sans modèle (mono, rééchantillonnage 16 kHz, niveau)",
    "vad": "Traitement du signal, sans modèle (seuil d'énergie adaptatif)",
    "features": "Extracteur log-Mel de Whisper (calcul fixe, sans apprentissage)",
    "codeswitch": "Règles sur l'écriture des mots, sans modèle",
    "normalize": "Règles d'écriture WRITING_NORM (notebook 03), sans modèle",
    "result": "Aucun modèle (assemblage et temps)",
}
_WHISPER_SIZES = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}


def describe_model(path_or_id: str, role: str, fallback_used: bool = False) -> str:
    """Nom lisible d'un modèle : architecture, taille, origine (SOURCE.txt écrit par les notebooks) ou repli."""
    p = Path(str(path_or_id))
    cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
    source = (p / "SOURCE.txt").read_text().strip() if (p / "SOURCE.txt").exists() else ""
    mtype = cfg.get("model_type", "")
    if fallback_used or not cfg:
        return f"{path_or_id} (modèle de base, zero-shot : REPLI, le modèle fine-tuné est absent)"
    if mtype == "whisper":
        size = _WHISPER_SIZES.get(cfg.get("d_model"), "")
        base = f"Whisper-{size}" if size else "Whisper"
        if role == "translate":
            tag = " (E2E-2)" if source.startswith("E2E-2") else ""
            return f"{base} en traduction directe{tag}, fine-tuné sur TEDxTN (encodeur-décodeur)"
        return f"{base} fine-tuné sur TEDxTN (encodeur-décodeur){f' · {source}' if source else ''}"
    if mtype == "m2m_100":
        size = "600M" if cfg.get("d_model") == 1024 and cfg.get("encoder_layers") == 12 else ""
        lora = " + LoRA" if "lora" in source else ""
        stage = f" ({source.split('-[', 1)[0]})" if "-[" in source else ""
        return f"NLLB-{size or '200'}{lora}{stage}, fine-tuné sur TEDxTN (traduction de texte)"
    if mtype in ("wav2vec2-bert", "wav2vec2"):
        return f"{mtype} + CTC fine-tuné sur TEDxTN (encodeur seul){f' · {source}' if source else ''}"
    return f"{p.name} ({mtype or 'modèle'})"


# ==========================================================================
# Backends réels (modèles des notebooks)
# ==========================================================================
def _model_kind(path_or_id: str) -> str:
    p = Path(path_or_id)
    cfg = p / "config.json"
    if cfg.exists():
        mt = json.loads(cfg.read_text()).get("model_type", "")
        return "whisper" if mt == "whisper" else "ctc"
    if (p / "adapter_config.json").exists():
        return "whisper"
    return "whisper" if "whisper" in path_or_id else "ctc"


class ASRBackend:
    """Charge le meilleur modèle ASR des notebooks, ou un modèle de base en repli."""

    def __init__(self, model_path: str, fallback: str, num_beams: int = 1):
        self.warning = ""
        if not Path(model_path).exists():
            self.warning = (f"Modèle fine-tuné introuvable ({model_path}). Repli sur {fallback} en zero-shot : "
                            f"les résultats seront nettement moins bons (cf. Table 6 du papier).")
            model_path = fallback
        self.path, self.kind, self.num_beams = model_path, _model_kind(model_path), num_beams
        self.label = describe_model(model_path, "asr", bool(self.warning))
        self.name = Path(model_path).name if Path(model_path).exists() else model_path
        if self.kind == "whisper":
            from .asr import DEVICE, load_whisper
            self.processor, self.model = load_whisper(model_path, task="transcribe", for_training=False)
            if DEVICE == "cuda":
                self.model = self.model.half()
            self.decoder = None
        else:
            from .asr import build_kenlm_decoder, load_ctc
            self.processor, self.model = load_ctc(model_path, None, from_checkpoint=True)
            lm = next((p for p in [LM_DIR / "tedxtn_4gram.bin", LM_DIR / "tedxtn_4gram.arpa"] if p.exists()), None)
            self.decoder = build_kenlm_decoder(self.processor, lm) if lm else None
            self.lm_name = lm.name if lm else None

    def describe(self) -> dict:
        d = {"modèle": self.name, "architecture": "Whisper encodeur-décodeur" if self.kind == "whisper"
             else "Encodeur auto-supervisé + CTC"}
        if self.kind == "whisper":
            d["décodage"] = f"beam search ({self.num_beams})" if self.num_beams > 1 else "glouton"
        else:
            d["décodage"] = f"beam + LM {self.lm_name}" if self.decoder else "glouton (sans LM)"
        return d

    def transcribe(self, wavs):
        if self.kind == "whisper":
            from .asr import whisper_generate
            return whisper_generate(self.model, self.processor, wavs, task="transcribe", num_beams=self.num_beams,
                                    return_details=True)
        from .asr import ctc_transcribe
        return ctc_transcribe(self.model, self.processor, wavs, decoder=self.decoder, return_details=True)


class E2EBackend:
    """Whisper fine-tuné en tâche `translate` : audio tunisien -> anglais, sans texte intermédiaire."""

    def __init__(self, model_path: str, fallback: str):
        self.warning = ""
        if not Path(model_path).exists():
            self.warning = f"Modèle AST fine-tuné introuvable ({model_path}). Repli sur {fallback} (zero-shot)."
            model_path = fallback
        from .asr import DEVICE, load_whisper
        self.name = Path(model_path).name if Path(model_path).exists() else model_path
        self.label = describe_model(model_path, "translate", bool(self.warning))
        self.processor, self.model = load_whisper(model_path, task="translate", for_training=False)
        if DEVICE == "cuda":
            self.model = self.model.half()

    def translate_audio(self, wavs):
        from .asr import whisper_generate
        return whisper_generate(self.model, self.processor, wavs, task="translate", num_beams=4, return_details=True)


class MTBackend:
    def __init__(self, model_path: str, fallback: str):
        self.warning = ""
        if not Path(model_path).exists():
            self.warning = (f"NLLB fine-tuné introuvable ({model_path}). NLLB de base utilisé "
                            f"({fallback}, langue source aeb_Arab).")
            model_path = fallback
        from .translate import load_nllb
        self.name = Path(model_path).name if Path(model_path).exists() else model_path
        self.label = describe_model(model_path, "mt", bool(self.warning))
        self.tok, self.model = load_nllb(model_path)

    def translate(self, texts, tgt_code, src_code=None):
        """src_code permet le pivot anglais (eng_Latn -> cible) pour le mode direct."""
        from .translate import nllb_translate
        old = self.tok.src_lang
        try:
            if src_code:
                self.tok.src_lang = src_code
            return nllb_translate(self.tok, self.model, texts, tgt_lang=tgt_code)
        finally:
            self.tok.src_lang = old


def _label(backend) -> str:
    """Nom du modèle d'un backend (backends de test : leur `name`)."""
    return getattr(backend, "label", "") or getattr(backend, "name", "") or type(backend).__name__


class LLMBackend:
    def __init__(self, model_id: str):
        from .llm import load_llm
        self.name = model_id
        self.label = f"{model_id.split('/')[-1]} (LLM instruct, quantifié en 4 bits, sans fine-tuning)"
        self.tok, self.model = load_llm(model_id)
        # Exemples (sortie ASR, référence) du TRAIN, écrits par le notebook 01 (section GenAI)
        from .config import DATA_DIR
        from .data import load_json
        self.fewshot = [tuple(x) for x in load_json(DATA_DIR / "llm_fewshot.json", [])]

    def correct(self, texts):
        from .llm import post_correct
        return post_correct(self.tok, self.model, texts, self.fewshot)

    def translate(self, texts, target_name):
        from .llm import llm_translate
        return llm_translate(self.tok, self.model, texts, target_name)


# ==========================================================================
# Pipeline
# ==========================================================================
EXPLAIN = {
    "input": "Le micro ou le fichier peut avoir n'importe quel format. Les modèles attendent un signal mono "
             "échantillonné à 16 kHz : on moyenne les canaux, on rééchantillonne, on retire la composante "
             "continue et on normalise le niveau crête.",
    "vad": "On repère les zones où quelqu'un parle (énergie au-dessus d'un seuil adaptatif) et on découpe "
           "l'audio en morceaux de 25 s maximum, aux silences. Whisper ne lit que 30 s à la fois ; découper aux "
           "pauses évite de couper un mot en deux.",
    "features": "Chaque morceau devient un spectrogramme log-Mel : 100 trames par seconde, 80 bandes de fréquence "
                "réparties comme l'oreille humaine. C'est cette image, et non l'onde brute, que lit l'encodeur.",
    "asr": "Le modèle acoustique fine-tuné sur TEDxTN convertit le spectrogramme en texte tunisien. Whisper génère "
           "des tokens un par un (couleur = confiance) ; un modèle CTC prédit un caractère par trame, puis les "
           "répétitions et les « blancs » sont fusionnés.",
    "codeswitch": "Le tunisien mélange arabe, français et anglais. Chaque mot est classé selon son script ; on en "
                  "déduit le type du segment (TUN / MIXED / FOR) et l'indice de mélange CMI (Das & Gambäck).",
    "normalize": "Le tunisien n'a pas d'orthographe standard. On applique des conventions homogènes (alef, ya, "
                 "diacritiques, allongements, frontière arabe/latin...) pour comparer et réutiliser le texte.",
    "llm": "Un LLM instruct relit la transcription et corrige les erreurs évidentes (mots mal entendus, emprunts "
           "dans le mauvais script) sans reformuler. Si sa sortie s'éloigne trop de l'entrée, on la rejette.",
    "translate": "Deux façons de traduire : en direct (end-to-end : l'audio va directement vers l'anglais) ou en "
                 "pipeline (texte reconnu -> traduction automatique NLLB, qui connaît le tunisien « aeb_Arab », "
                 "ou un LLM). En mode comparaison, on lance les deux et on mesure leur accord.",
    "result": "Texte final et temps passé dans chaque étape.",
}

TITLES = {"input": "Signal d'entrée", "vad": "Détection de parole", "features": "Représentation acoustique",
          "asr": "Reconnaissance (ASR)", "codeswitch": "Code-switching", "normalize": "Normalisation",
          "llm": "Post-correction LLM", "translate": "Traduction", "result": "Résultat"}


class TunisianSpeechPipeline:
    def __init__(self, models: AppModels | None = None, asr=None, e2e=None, mt=None, llm=None, num_beams: int = 1):
        """Les backends peuvent être injectés (tests) ; sinon ils sont chargés à la demande."""
        self.models = models or AppModels()
        self._asr, self._e2e, self._mt, self._llm = asr, e2e, mt, llm
        self.num_beams = num_beams

    # chargement paresseux : l'interface démarre vite, chaque modèle n'est chargé qu'au premier usage
    @property
    def asr(self):
        if self._asr is None:
            self._asr = ASRBackend(self.models.asr, self.models.asr_fallback, self.num_beams)
        return self._asr

    @property
    def e2e(self):
        if self._e2e is None:
            self._e2e = E2EBackend(self.models.ast_e2e, self.models.ast_e2e_fallback)
        return self._e2e

    @property
    def mt(self):
        if self._mt is None:
            self._mt = MTBackend(self.models.mt, self.models.mt_fallback)
        return self._mt

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LLMBackend(self.models.llm)
        return self._llm

    # ------------------------------------------------------------------
    def run(self, audio, sr: int | None = None, **kw) -> PipelineResult:
        res = None
        for res in self.run_iter(audio, sr, **kw):
            pass
        return res

    def run_iter(self, audio, sr: int | None = None, target: str = "Anglais", mode: str = "pipeline",
                 use_llm: bool = False, mt_engine: str = "NLLB") -> Iterator[PipelineResult]:
        """Exécute le pipeline et renvoie le résultat partiel après chaque étape (affichage progressif).
        mode : 'direct' (end-to-end), 'pipeline' (ASR -> MT) ou 'compare' (les deux)."""
        res = PipelineResult(target=target)
        t_all = time.perf_counter()

        def add(key, fn):
            st = Step(key, TITLES[key], EXPLAIN[key], model=STEP_MODELS.get(key, ""))
            t0 = time.perf_counter()
            try:
                fn(st)
            except Exception as e:  # une étape qui échoue ne fait pas tomber l'interface
                st.status, st.summary = "error", f"Erreur : {type(e).__name__}: {e}"
                st.html = f"<pre class='err'>{html.escape(repr(e))}</pre>"
            st.seconds = time.perf_counter() - t0
            res.steps.append(st)
            return st

        # ---------------- 1. Signal ----------------
        state: dict = {}

        def s_input(st):
            raw = audio
            if isinstance(raw, tuple):  # format Gradio (sr, ndarray)
                sr_in, arr = raw
            else:
                sr_in, arr = sr, raw
            if isinstance(arr, (str, Path)):
                import soundfile as sf
                info = sf.info(str(arr))
                n_ch, sr_in = info.channels, info.samplerate
                wav = load_audio(arr)
            else:
                arr = np.asarray(arr)
                n_ch = 1 if arr.ndim == 1 else min(arr.shape)
                wav = load_audio(arr, sr_in=sr_in)
            wav = wav - wav.mean()
            peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
            clip = float(np.mean(np.abs(wav) > 0.99) * 100) if len(wav) else 0.0
            rms_db = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-10) if len(wav) else -100
            if peak > 0:
                wav = wav / peak * 0.9
            state["wav"] = wav
            dur = len(wav) / SAMPLING_RATE
            if dur < 0.3:
                raise ValueError("Audio trop court (moins de 0,3 s).")
            st.summary = f"{dur:.1f} s · {sr_in} Hz → 16 kHz · {n_ch} canal(aux) → mono"
            st.data = {"durée (s)": round(dur, 2), "fréquence d'origine": f"{sr_in} Hz", "canaux d'origine": n_ch,
                       "niveau RMS": f"{rms_db:.1f} dBFS", "crête": f"{peak:.2f}", "écrêtage": f"{clip:.2f} %"}
            if rms_db < -45:
                st.status, st.summary = "warn", st.summary + " · signal très faible"
            if clip > 1:
                st.status, st.summary = "warn", st.summary + " · saturation détectée"
            st.html = kv_html(st.data)
            st.figure = waveform_figure(wav, SAMPLING_RATE, title="Forme d'onde normalisée")

        add("input", s_input)
        yield res
        if "wav" not in state:
            res.total_seconds = time.perf_counter() - t_all
            return

        wav = state["wav"]

        # ---------------- 2. VAD ----------------
        def s_vad(st):
            regions, chunks, e, thr = energy_vad(wav)
            state["chunks"] = chunks
            speech_s = sum(b - a for a, b in regions)
            st.summary = f"{len(regions)} zone(s) de parole · {len(chunks)} morceau(x) · {100 * speech_s / (len(wav) / SAMPLING_RATE):.0f} % de parole"
            st.data = {"seuil d'énergie": f"{thr:.1f} dB", "zones de parole": len(regions), "morceaux": len(chunks),
                       "temps de parole": f"{speech_s:.1f} s"}
            st.html = kv_html(st.data) + "<div class='chips'>" + "".join(
                f"<span class='chip'>#{i + 1} · {a:.1f}–{b:.1f} s</span>" for i, (a, b) in enumerate(chunks)) + "</div>"
            st.figure = waveform_figure(wav, SAMPLING_RATE, regions, chunks,
                                        title="Zones de parole (vert) et découpage (pointillés)")

        add("vad", s_vad)
        yield res
        chunks = state.get("chunks", [(0.0, len(wav) / SAMPLING_RATE)])
        pieces = [wav[int(a * SAMPLING_RATE):int(b * SAMPLING_RATE)] for a, b in chunks]

        # ---------------- 3. Features ----------------
        def s_feat(st):
            mel = log_mel(pieces[0])
            st.summary = f"{mel.shape[1]} trames × {mel.shape[0]} bandes Mel (morceau #1)"
            st.data = {"fenêtre": "25 ms", "pas": "10 ms", "bandes Mel": mel.shape[0],
                       "trames (morceau #1)": mel.shape[1]}
            st.html = kv_html(st.data)
            st.figure = spectrogram_figure(mel)

        add("features", s_feat)
        yield res

        # ---------------- 4. ASR ----------------
        def s_asr(st):
            backend = self.asr
            st.model = _label(backend)
            outs = backend.transcribe(pieces)
            texts = [o["text"] if isinstance(o, dict) else o for o in outs]
            res.transcript = " ".join(t for t in texts if t).strip()
            confs = [o.get("confidence", 0) for o in outs if isinstance(o, dict)]
            st.data = {**backend.describe(), "morceaux": len(pieces),
                       "confiance moyenne": f"{100 * np.mean(confs):.0f} %" if confs else "n/d"}
            detail = []
            for i, o in enumerate(outs):
                if isinstance(o, dict) and "tokens" in o:
                    detail.append(f"<div class='chunk-h'>Morceau #{i + 1}</div>" + confidence_html(o["tokens"], o["token_probs"]))
                elif isinstance(o, dict) and "frames" in o:
                    detail.append(f"<div class='chunk-h'>Morceau #{i + 1} · émissions CTC (1 case = 20 ms)</div>"
                                  + ctc_frames_html(o["frames"], backend.processor.tokenizer.pad_token)
                                  + f"<div class='chunk-text' dir='auto'>{html.escape(o['text'])}</div>")
            legend = ("<div class='legend'><span class='tok tok-hi'>sûr</span><span class='tok tok-mid'>moyen</span>"
                      "<span class='tok tok-lo'>incertain</span></div>")
            st.html = kv_html(st.data) + legend + "".join(detail)
            st.summary = f"{len(res.transcript.split())} mots · {st.data['confiance moyenne']} de confiance"
            if getattr(backend, "warning", ""):
                st.status = "warn"
                st.html = f"<div class='warn'>{html.escape(backend.warning)}</div>" + st.html

        # La transcription est calculée dans tous les modes (même en direct) pour l'analyse.
        add("asr", s_asr)
        yield res

        # ---------------- 5. Code-switching ----------------
        def s_cs(st):
            t = res.transcript
            stats = code_switch_stats([t])
            st.data = {"type de segment": segment_type(t), "CMI": f"{cmi(t):.1f} %",
                       "mots tunisiens (arabe)": stats["words_ar"], "mots étrangers (latin)": stats["words_foreign"],
                       "intra-mot": stats["words_intra_cs"], "hésitations": stats["hesitations"]}
            st.summary = f"{st.data['type de segment']} · CMI {st.data['CMI']} · {stats['words_foreign']} mot(s) étranger(s)"
            legend = ("<div class='legend'><span class='cs-tok cs-ar' style='--c:#0f766e'>tunisien</span>"
                      "<span class='cs-tok cs-lat' style='--c:#b45309'>français/anglais</span>"
                      "<span class='cs-tok cs-mixed' style='--c:#7c3aed'>intra-mot</span>"
                      "<span class='cs-tok cs-hes' style='--c:#64748b'>hésitation</span></div>")
            st.html = kv_html(st.data) + legend + code_switch_html(t)

        if res.transcript:
            add("codeswitch", s_cs)
            yield res

        # ---------------- 6. Normalisation ----------------
        def s_norm(st):
            norm, trace = normalize_text(res.transcript, WRITING_NORM, return_trace=True)   # convention d'écriture (notebook 03)
            res.transcript_normalized = norm
            st.summary = f"{len(trace)} règle(s) appliquée(s)" if trace else "aucune règle nécessaire"
            items = "".join(f"<li><b dir='ltr'>{_ltr_label(t['rule'])}</b>{diff_html(t['before'], t['after'])}</li>" for t in trace)
            st.html = (f"<ol class='rules'>{items}</ol>" if items else "<p>Le texte respecte déjà les conventions.</p>") + \
                      f"<div class='final-norm' dir='auto'>{html.escape(norm)}</div>"

        if res.transcript:
            add("normalize", s_norm)
            yield res

        # ---------------- 7. LLM ----------------
        mt_input = res.transcript

        def s_llm(st):
            nonlocal mt_input
            st.model = _label(self.llm)
            corrected, accepted = self.llm.correct([res.transcript])
            mt_input = corrected[0]
            st.summary = "correction acceptée" if accepted[0] else "aucune correction retenue"
            st.data = {"modèle": self.llm.name}
            st.status = "warn"
            st.html = f"<div class='note'>{html.escape(LLM_CORRECTION_WARNING)}</div>" + kv_html(st.data) + \
                      diff_html(res.transcript, corrected[0])

        if use_llm and res.transcript:
            add("llm", s_llm)
            yield res

        # ---------------- 8. Traduction ----------------
        info = TARGET_LANGUAGES[target]

        def s_tr(st):
            rows, notes, used = [], [], []
            if mode in ("direct", "compare"):
                o = self.e2e.translate_audio(pieces)
                used.append(_label(self.e2e))
                en = " ".join((x["text"] if isinstance(x, dict) else x) for x in o).strip()
                if info["whisper"] == "en":
                    res.translations["Directe (end-to-end)"] = en
                else:
                    # le modèle E2E n'est entraîné que vers l'anglais (seules références de TEDxTN) : pivot
                    piv = self.mt.translate([en], info["nllb"], src_code="eng_Latn")[0]
                    used.append(_label(self.mt) + " (pivot anglais)")
                    res.translations["Directe + pivot anglais"] = piv
                    notes.append("Le modèle end-to-end ne produit que de l'anglais (seule langue de référence de "
                                 "TEDxTN). Vers une autre langue, l'anglais sert de pivot (NLLB eng_Latn → cible).")
                if getattr(self.e2e, "warning", ""):
                    notes.append(self.e2e.warning)
            if mode in ("pipeline", "compare") and mt_input:
                if mt_engine in ("NLLB", "NLLB + LLM"):
                    res.translations["Pipeline ASR → NLLB"] = self.mt.translate([mt_input], info["nllb"])[0]
                    used.append(_label(self.mt))
                    if getattr(self.mt, "warning", ""):
                        notes.append(self.mt.warning)
                if mt_engine in ("LLM", "NLLB + LLM"):
                    res.translations["Pipeline ASR → LLM"] = self.llm.translate([mt_input], info["llm_name"])[0]
                    used.append(_label(self.llm))
                    notes.append(LLM_TRANSLATION_WARNING)
            for k, v in res.translations.items():
                rows.append(f"<div class='tr-card'><div class='tr-h'>{html.escape(k)}</div>"
                            f"<div class='tr-t' dir='auto'>{html.escape(v)}</div></div>")
            agree = ""
            if len(res.translations) >= 2:
                import sacrebleu
                vals = list(res.translations.values())
                sc = sacrebleu.sentence_chrf(vals[0], [vals[1]]).score
                agree = f"<div class='agree'>Accord entre les deux premières traductions (chrF) : <b>{sc:.0f}</b>/100</div>"
                st.data["accord chrF"] = round(sc, 1)
            st.model = " + ".join(dict.fromkeys(used))
            st.summary = f"{target} · " + ", ".join(res.translations) if res.translations else "aucune traduction"
            st.html = "".join(f"<div class='note'>{html.escape(n)}</div>" for n in notes) + \
                      f"<div class='tr-grid'>{''.join(rows)}</div>" + agree
            if notes:
                st.status = "warn" if any("introuvable" in n or n == LLM_TRANSLATION_WARNING for n in notes) else st.status

        add("translate", s_tr)
        yield res

        # ---------------- 9. Résultat ----------------
        res.total_seconds = time.perf_counter() - t_all

        def s_res(st):
            st.summary = f"terminé en {res.total_seconds:.1f} s"
            st.figure = timing_figure(res.steps)

        add("result", s_res)
        yield res
