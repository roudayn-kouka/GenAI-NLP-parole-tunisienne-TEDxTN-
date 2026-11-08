"""
Métriques d'évaluation et outils d'analyse d'erreurs.

ASR  : WER, CER (bruts et normalisés), WER par type de segment (TUN/MIXED/FOR),
       décomposition substitutions / suppressions / insertions,
       classification automatique des substitutions (script, orthographe, emprunt...).
AST  : BLEU (protocole du papier + standard), chrF++, COMET (optionnel),
       test de significativité par bootstrap apparié.
Tous : intervalles de confiance à 95 % par bootstrap.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .text import (EVAL_NORM, RAW_NORM, NormConfig, PUNCT, is_script_variant, is_spelling_variant,
                   normalize_text, segment_type, token_script)


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------
def _prep(refs, hyps, cfg: NormConfig | None):
    refs = [normalize_text(r, cfg) if cfg else str(r) for r in refs]
    hyps = [normalize_text(h, cfg) if cfg else str(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r.strip()]
    return [refs[i] for i in keep], [hyps[i] if hyps[i].strip() else "<vide>" for i in keep], keep


def wer_cer(refs: Sequence[str], hyps: Sequence[str], cfg: NormConfig | None = RAW_NORM) -> dict:
    """WER et CER (en %) avec décomposition des erreurs.
    `cfg=RAW_NORM` reproduit le protocole du papier (ponctuation et casse ignorées) ;
    `cfg=EVAL_NORM` donne le WER normalisé (variations orthographiques neutralisées)."""
    import jiwer
    r, h, keep = _prep(refs, hyps, cfg)
    if not r:
        return {"WER": float("nan"), "CER": float("nan"), "n": 0}
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    n_ref = w.hits + w.substitutions + w.deletions
    return {
        "WER": 100 * w.wer, "CER": 100 * c.cer,
        "sub%": 100 * w.substitutions / n_ref, "del%": 100 * w.deletions / n_ref,
        "ins%": 100 * w.insertions / n_ref,
        "n": len(r), "ref_words": n_ref, "skipped_empty_refs": len(refs) - len(keep),
    }


def segment_wer(ref: str, hyp: str, cfg: NormConfig | None = RAW_NORM) -> float:
    """WER d'un seul segment (en %), robuste aux chaînes vides."""
    import jiwer
    r = normalize_text(ref, cfg) if cfg else ref
    h = normalize_text(hyp, cfg) if cfg else hyp
    if not r.strip():
        return 0.0 if not h.strip() else 100.0
    return 100 * jiwer.wer(r, h if h.strip() else "<vide>")


def wer_by_segment_type(refs, hyps, cfg: NormConfig | None = RAW_NORM) -> pd.DataFrame:
    """Reproduit la Table 9 du papier : WER sur les sous-ensembles TUN / MIXED / FOR."""
    types = [segment_type(r) for r in refs]
    rows = []
    for t in ["TUN", "MIXED", "FOR", "ALL"]:
        idx = [i for i, x in enumerate(types) if t == "ALL" or x == t]
        if not idx:
            continue
        m = wer_cer([refs[i] for i in idx], [hyps[i] for i in idx], cfg)
        rows.append({"type": t, "segments": len(idx), "mots_ref": m["ref_words"],
                     "WER": m["WER"], "CER": m["CER"]})
    return pd.DataFrame(rows)


def classify_substitution(ref_w: str, hyp_w: str) -> str:
    """Catégorie d'une substitution mot à mot."""
    if is_spelling_variant(ref_w, hyp_w):
        return "variante orthographique"
    if is_script_variant(ref_w, hyp_w):
        return "même mot, autre script"
    sr, sh = token_script(ref_w), token_script(hyp_w)
    if "lat" in (sr, sh) or "mixed" in (sr, sh):
        return "mot emprunté / code-switching"
    from difflib import SequenceMatcher
    if SequenceMatcher(None, ref_w, hyp_w).ratio() >= 0.6:
        return "proche phonétiquement (prononciation)"
    return "autre (lexical)"


def error_table(refs, hyps, cfg: NormConfig | None = RAW_NORM) -> pd.DataFrame:
    """Liste toutes les opérations d'erreur (S/D/I) avec leur catégorie.
    Sert à l'analyse qualitative du notebook 01."""
    import jiwer
    r, h, keep = _prep(refs, hyps, cfg)
    out = jiwer.process_words(r, h)
    rows = []
    for seg_i, (al, rw, hw) in enumerate(zip(out.alignments, out.references, out.hypotheses)):
        for chunk in al:
            if chunk.type == "equal":
                continue
            rseg = rw[chunk.ref_start_idx:chunk.ref_end_idx]
            hseg = hw[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if chunk.type == "substitute":
                for a, b in zip(rseg, hseg):
                    rows.append({"seg": keep[seg_i], "op": "S", "ref": a, "hyp": b,
                                 "categorie": classify_substitution(a, b)})
            elif chunk.type == "delete":
                for a in rseg:
                    rows.append({"seg": keep[seg_i], "op": "D", "ref": a, "hyp": "",
                                 "categorie": "suppression (" + token_script(a) + ")"})
            elif chunk.type == "insert":
                for b in hseg:
                    rows.append({"seg": keep[seg_i], "op": "I", "ref": "", "hyp": b,
                                 "categorie": "insertion (" + token_script(b) + ")"})
    return pd.DataFrame(rows, columns=["seg", "op", "ref", "hyp", "categorie"])


def script_aware_wer(refs, hyps, cfg: NormConfig = EVAL_NORM) -> dict:
    """WER où une substitution « même mot, autre script » n'est pas comptée comme erreur.
    Donne une borne optimiste qui quantifie le biais d'orthographe signalé par le papier."""
    base = wer_cer(refs, hyps, cfg)
    errs = error_table(refs, hyps, cfg)
    forgiven = int(((errs["op"] == "S") & (errs["categorie"] == "même mot, autre script")).sum()) if len(errs) else 0
    n_ref = base["ref_words"]
    wer_sa = base["WER"] - 100 * forgiven / n_ref if n_ref else float("nan")
    return {"WER_normalise": base["WER"], "WER_script_aware": wer_sa, "substitutions_pardonnees": forgiven}


# --------------------------------------------------------------------------
# Traduction
# --------------------------------------------------------------------------
def _strip_punct(s: str) -> str:
    return " ".join(PUNCT.sub(" ", str(s)).split())


def translation_scores(hyps: Sequence[str], refs: Sequence[str], sources: Sequence[str] | None = None,
                       comet_model=None, comet_batch_size: int = 32) -> dict:
    """BLEU (protocole du papier : true-case, sans ponctuation), BLEU standard (sacrebleu 13a),
    chrF++ et, si un modèle COMET est fourni, le score COMET (x100)."""
    import sacrebleu
    hyps = [str(h) for h in hyps]
    refs = [str(r) for r in refs]
    res = {
        "BLEU": sacrebleu.corpus_bleu([_strip_punct(h) for h in hyps], [[_strip_punct(r) for r in refs]]).score,
        "BLEU_sacre": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "chrF++": sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
        "n": len(hyps),
    }
    if comet_model is not None and sources is not None:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
        out = comet_model.predict(data, batch_size=comet_batch_size, gpus=1 if _has_cuda() else 0,
                                  progress_bar=False)
        res["COMET"] = 100 * float(out.system_score)
        res["_comet_segments"] = [float(x) for x in out.scores]
    return res


def sentence_chrf(hyp: str, ref: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(str(hyp), [str(ref)], word_order=2).score


def paired_bootstrap(metric_fn: Callable[[list, list], float], hyps_a, hyps_b, refs,
                     n_samples: int = 1000, seed: int = 42) -> dict:
    """Test de significativité par bootstrap apparié (Koehn, 2004).
    Renvoie la différence moyenne B - A et la p-valeur que B ne soit pas meilleur que A."""
    rng = random.Random(seed)
    n = len(refs)
    diffs = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        a = metric_fn([hyps_a[i] for i in idx], [refs[i] for i in idx])
        b = metric_fn([hyps_b[i] for i in idx], [refs[i] for i in idx])
        diffs.append(b - a)
    diffs = np.array(diffs)
    return {"delta_moyen": float(diffs.mean()), "IC95": (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5))), "p_value": float((diffs <= 0).mean())}


# --------------------------------------------------------------------------
# Intervalles de confiance
# --------------------------------------------------------------------------
def bootstrap_ci(metric_fn: Callable[[list, list], float], hyps, refs, n_samples: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Intervalle de confiance bootstrap (rééchantillonnage des segments).
    Important ici : dev et test ne contiennent que 5 et 6 talks, les scores sont instables."""
    rng = random.Random(seed)
    n = len(refs)
    vals = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(metric_fn([hyps[i] for i in idx], [refs[i] for i in idx]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def wer_fn(cfg: NormConfig | None = RAW_NORM):
    """Fabrique une fonction (hyps, refs) -> WER, utilisable avec bootstrap_ci."""
    return lambda h, r: wer_cer(r, h, cfg)["WER"]


def bleu_fn():
    import sacrebleu
    return lambda h, r: sacrebleu.corpus_bleu([_strip_punct(x) for x in h], [[_strip_punct(x) for x in r]]).score


def chrf_fn():
    import sacrebleu
    return lambda h, r: sacrebleu.corpus_chrf(list(h), [list(r)], word_order=2).score


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------
# Catégorisation des erreurs de traduction
# --------------------------------------------------------------------------
def translation_error_flags(src: str, hyp: str, ref: str) -> list[str]:
    """Heuristiques pour pré-classer les erreurs de traduction avant l'analyse manuelle."""
    flags = []
    h, r = str(hyp).split(), str(ref).split()
    if len(r) and len(h) / len(r) < 0.6:
        flags.append("omission (trop court)")
    if len(r) and len(h) / len(r) > 1.6:
        flags.append("ajout / hallucination (trop long)")
    if h:
        counts = Counter(h)
        if counts.most_common(1)[0][1] >= 4 and counts.most_common(1)[0][1] / len(h) > 0.3:
            flags.append("répétition en boucle")
    ar_left = sum(token_script(w) in {"ar", "mixed"} for w in h)
    if ar_left:
        flags.append("mots non traduits (arabe)")
    fr_markers = {"le", "la", "les", "des", "est", "une", "pour", "avec", "dans", "que", "je", "c'est"}
    if sum(w.lower() in fr_markers for w in h) >= 2:
        flags.append("français non traduit (code-switching)")
    if sentence_chrf(hyp, ref) < 25 and not flags:
        flags.append("contresens / sens perdu")
    return flags or ["correcte ou mineure"]
