"""Comparaison d'une analyse de l'interface avec la référence d'un segment du corpus (onglet « Voix existantes »).

Démonstration seulement : mêmes conventions que l'évaluation des notebooks (WER / CER avec RAW_NORM, protocole du
papier ; BLEU phrase sans ponctuation, casse gardée ; chrF++), sur un seul segment. Aucun système ne change.
"""
from __future__ import annotations

import html

from common import metrics
from common.pipeline import diff_html
from common.text import RAW_NORM, normalize_text


def _strip(s: str) -> str:
    return metrics._strip_punct(str(s))


def compare_to_reference(transcript: str, translations: dict, ref_transcript: str, ref_translation: str) -> dict:
    """Scores d'un segment : WER / CER de la transcription, BLEU / chrF++ de chaque traduction anglaise."""
    import sacrebleu
    wc = metrics.wer_cer([ref_transcript], [transcript or ""], RAW_NORM)
    out = {"WER": round(wc["WER"], 2), "CER": round(wc["CER"], 2), "translations": {}}
    for name, hyp in (translations or {}).items():
        out["translations"][name] = {
            "BLEU": round(sacrebleu.sentence_bleu(_strip(hyp), [_strip(ref_translation)]).score, 2),
            "chrF++": round(sacrebleu.sentence_chrf(str(hyp), [str(ref_translation)], word_order=2).score, 2),
        }
    return out


def reference_html(scores: dict, transcript: str, translations: dict, ref_transcript: str, ref_translation: str) -> str:
    """Carte « comparaison avec la référence » : scores et diff mot à mot (référence -> sortie du système)."""
    a = normalize_text(ref_transcript, RAW_NORM)
    b = normalize_text(transcript or "", RAW_NORM)
    tiles = (f"<div class='dv-score'><span>WER</span><b>{scores['WER']:.1f} %</b></div>"
             f"<div class='dv-score'><span>CER</span><b>{scores['CER']:.1f} %</b></div>")
    for name, sc in scores["translations"].items():
        tiles += (f"<div class='dv-score'><span>{html.escape(name)}</span>"
                  f"<b>BLEU {sc['BLEU']:.1f} · chrF++ {sc['chrF++']:.1f}</b></div>")
    tr_diffs = "".join(
        f"<div class='dv-sub'>{html.escape(name)}</div>{diff_html(ref_translation, hyp)}"
        for name, hyp in (translations or {}).items())
    return (
        "<div class='dv-card dv-ref'>"
        "<div class='dv-label'>Comparaison avec la référence du corpus</div>"
        f"<div class='dv-scores'>{tiles}</div>"
        "<p class='dv-hint'>Scores sur ce seul segment (démonstration) : WER / CER plus bas = mieux, BLEU / chrF++ plus "
        "haut = mieux. Barré : mot de la référence absent de la sortie ; surligné : mot ajouté ou différent.</p>"
        "<div class='dv-sub'>Transcription (référence → sortie)</div>"
        f"{diff_html(a, b)}"
        f"<div class='dv-sub'>Référence anglaise</div><div class='dv-reftext' dir='ltr'>{html.escape(ref_translation)}</div>"
        f"{tr_diffs}</div>")
