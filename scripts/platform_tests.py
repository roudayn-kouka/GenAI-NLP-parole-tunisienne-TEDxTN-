"""Étape 6 : test de la plateforme (TunisianSpeechPipeline, celle de app/app.py) sur de vrais segments du test.

    python scripts/platform_tests.py                  # 20 segments du test x 3 modes x 3 langues cibles
    python scripts/platform_tests.py --n 30
    python scripts/platform_tests.py --fake --split train   # essai à blanc, sans modèle ni GPU (backends factices)

À lancer quand les modèles finaux existent (models/asr_best, ast_best, mt_best ; sinon l'interface prend les modèles
de repli du Hub, signalé dans la colonne « avertissements »). C'est un test de fonctionnement, pas une évaluation :
statut de chaque étape, temps, sorties produites ; pour l'anglais (seule langue avec référence), chrF++ de la phrase
à titre indicatif. Segments tirés au hasard (graine du projet), stratifiés par type (TUN / MIXED / FOR).
Sortie : results/platform_tests.csv (une ligne par segment x mode x langue).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, metrics  # noqa: E402
from common.config import RESULTS_DIR, SAMPLING_RATE, SEED, TARGET_LANGUAGES  # noqa: E402
from common.pipeline import TunisianSpeechPipeline  # noqa: E402

MODES = ["pipeline", "direct", "compare"]


def pick(df, n):
    """n segments stratifiés par type de segment (au moins un de chaque type présent), de 1 à 20 s."""
    d = df[df.duration.between(1, 20)]
    types = sorted(d.seg_type.unique())
    per = {t: max(1, round(n * (d.seg_type == t).mean())) for t in types}
    while sum(per.values()) > n:
        per[max(per, key=per.get)] -= 1
    out = pd.concat([d[d.seg_type == t].sample(min(k, (d.seg_type == t).sum()), random_state=SEED) for t, k in per.items()])
    return out.sort_values("seg_id").reset_index(drop=True)


def fake_pipeline():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_core import FakeASR, FakeE2E, FakeMT
    return TunisianSpeechPipeline(asr=FakeASR(), e2e=FakeE2E(), mt=FakeMT())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fake", action="store_true", help="backends factices (essai du script, sans modèle)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RESULTS_DIR / ("platform_tests.csv" if not a.fake else "platform_tests_fake.csv")

    df = data.cut_segments(data.load_corpus(download=False))
    seg = pick(df[df.split == a.split], a.n)
    print(f"{len(seg)} segments ({a.split}) :", seg.seg_type.value_counts().to_dict())
    pipe = fake_pipeline() if a.fake else TunisianSpeechPipeline()
    rows = []
    for r in seg.itertuples():
        wav = data.load_audio(r.path)
        for mode in MODES:
            for target in TARGET_LANGUAGES:
                t0 = time.perf_counter()
                try:
                    res = pipe.run((SAMPLING_RATE, wav), target=target, mode=mode)
                    err = ""
                except Exception as e:   # une erreur non rattrapée par l'interface est un échec du test
                    res, err = None, f"{type(e).__name__}: {e}"
                steps = res.steps if res else []
                tr = res.translations if res else {}
                row = {"seg_id": r.seg_id, "talk": r.talk, "seg_type": r.seg_type, "durée_s": round(r.duration, 2),
                       "mode": mode, "cible": target, "ok": bool(res) and not err and all(s.status != "error" for s in steps),
                       "étapes en erreur": "; ".join(f"{s.key}: {s.summary}" for s in steps if s.status == "error") or err,
                       "avertissements": "; ".join(f"{s.key}: {s.summary}" for s in steps if s.status == "warn"),
                       "temps_s": round(time.perf_counter() - t0, 2), "transcription": res.transcript if res else "",
                       "référence": r.transcription, "n_traductions": len(tr),
                       "traductions": " || ".join(f"{k}: {v}" for k, v in tr.items())}
                if TARGET_LANGUAGES[target].get("has_reference") and tr and str(r.translation).strip():
                    row["chrF++ (indicatif)"] = round(np.mean([metrics.sentence_chrf(v, r.translation) for v in tr.values()]), 1)
                rows.append(row)
        print(f"  {r.seg_id} : {sum(x['ok'] for x in rows if x['seg_id'] == r.seg_id)}/{len(MODES) * len(TARGET_LANGUAGES)} ok")
    tab = pd.DataFrame(rows)
    tab.to_csv(out, index=False)
    print(f"\n{tab.ok.sum()}/{len(tab)} exécutions sans erreur ; temps médian {tab.temps_s.median():.1f} s ->", out)
    print(tab.groupby(["mode", "cible"]).agg(ok=("ok", "mean"), temps_s=("temps_s", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
