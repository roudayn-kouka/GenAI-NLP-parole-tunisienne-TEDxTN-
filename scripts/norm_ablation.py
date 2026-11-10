"""Ablation de la normalisation EVAL_NORM, règle par règle, sur des prédictions du DEV.

    python scripts/norm_ablation.py                                   # results/preds/asr__whisper-small-ft__dev.csv
    python scripts/norm_ablation.py --preds results/preds/asr__<système>__dev.csv

Pour chaque règle qui distingue EVAL_NORM de RAW_NORM :
  - « seule » : RAW_NORM + cette règle (effet isolé) ;
  - « retirée » : EVAL_NORM sans cette règle (effet dans le profil complet).
On rapporte WER, nombre d'erreurs (S+D+I) et nombre de mots de référence : le WER est un rapport, une règle peut le
faire monter en retirant surtout des mots bien reconnus (dénominateur) ou en créant des erreurs (numérateur).
Exemples : segments où la règle seule ajoute des erreurs par rapport à RAW_NORM.
Sorties : results/norm_ablation_dev.csv, results/norm_ablation_dev_examples.csv. Refuse un fichier de test.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import RESULTS_DIR  # noqa: E402
from common.text import EVAL_NORM, RAW_NORM, normalize_text  # noqa: E402


def seg_counts(refs, hyps, cfg):
    """Erreurs (S+D+I) et mots de référence par segment, après normalisation (mêmes conventions que metrics._prep)."""
    import jiwer
    out = []
    for r, h in zip(refs, hyps):
        rn, hn = normalize_text(r, cfg), normalize_text(h, cfg)
        if not rn.strip():
            out.append((0, 0, rn, hn)); continue
        hn = hn if hn.strip() else "<vide>"
        w = jiwer.process_words(rn, hn)
        out.append((w.substitutions + w.deletions + w.insertions, w.hits + w.substitutions + w.deletions, rn, hn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(RESULTS_DIR / "preds" / "asr__whisper-small-ft__dev.csv"))
    ap.add_argument("--n-examples", type=int, default=5)
    a = ap.parse_args()
    assert "__test" not in Path(a.preds).name, "ablation réservée au dev (aucune décision sur le test)"
    p = pd.read_csv(a.preds, keep_default_na=False)
    refs, hyps = p.ref.tolist(), p.hyp.tolist()

    rules = [k for k in vars(EVAL_NORM) if getattr(EVAL_NORM, k) != getattr(RAW_NORM, k)]
    base = seg_counts(refs, hyps, RAW_NORM)
    full = seg_counts(refs, hyps, EVAL_NORM)

    def tot(c):
        e, n = sum(x[0] for x in c), sum(x[1] for x in c)
        return e, n, 100 * e / n

    rows, ex = [], []
    for name, c in [("RAW_NORM", base), ("EVAL_NORM", full)]:
        e, n, w = tot(c); rows.append({"config": name, "règle": "", "WER": w, "erreurs": e, "mots_ref": n})
    e0, n0, w0 = tot(base); e1, n1, w1 = tot(full)
    for r in rules:
        alone = seg_counts(refs, hyps, replace(RAW_NORM, **{r: getattr(EVAL_NORM, r)}))
        wo = seg_counts(refs, hyps, replace(EVAL_NORM, **{r: getattr(RAW_NORM, r)}))
        ea, na, wa = tot(alone); ew, nw, ww = tot(wo)
        added = [i for i in range(len(p)) if alone[i][0] > base[i][0]]
        removed = [i for i in range(len(p)) if alone[i][0] < base[i][0]]
        rows.append({"config": "RAW + règle seule", "règle": r, "WER": wa, "ΔWER": wa - w0, "erreurs": ea, "Δerreurs": ea - e0,
                     "mots_ref": na, "Δmots_ref": na - n0, "segments +erreurs": len(added), "segments -erreurs": len(removed)})
        rows.append({"config": "EVAL sans la règle", "règle": r, "WER": ww, "ΔWER": ww - w1, "erreurs": ew, "Δerreurs": ew - e1,
                     "mots_ref": nw, "Δmots_ref": nw - n1})
        for i in sorted(added, key=lambda i: alone[i][0] - base[i][0], reverse=True)[: a.n_examples]:
            ex.append({"règle": r, "seg_id": p.seg_id[i], "erreurs_raw": base[i][0], "erreurs_règle": alone[i][0],
                       "ref_raw": base[i][2], "hyp_raw": base[i][3], "ref_règle": alone[i][2], "hyp_règle": alone[i][3]})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 200)
    print(f"{Path(a.preds).name} : {len(p)} segments\n")
    print(tab.round(2).to_string(index=False))
    tab.to_csv(RESULTS_DIR / "norm_ablation_dev.csv", index=False)
    pd.DataFrame(ex).to_csv(RESULTS_DIR / "norm_ablation_dev_examples.csv", index=False)


if __name__ == "__main__":
    main()
