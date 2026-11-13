"""Enregistre dans results/ast_paired_test.csv le test apparié (bootstrap, 1 000 tirages, graine 42) entre le meilleur
pipeline et le meilleur système direct choisis sur le dev, calculé comme le notebook 02 (cellule 29) à partir des
prédictions du test. Aucun système ne change ; sert à l'onglet « Résultats » de l'interface (chiffres lus dans un CSV).

    python scripts/paired_test_csv.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import pandas as pd

from common import data, metrics

dev = pd.read_csv("results/ast_results_dev.csv")
direct = dev[dev.type == "direct"].iloc[0]["système"]
pipe = dev[dev.type == "pipeline"].iloc[0]["système"]
A = pd.read_csv(data.pred_path("ast", direct, "test")).fillna("")
B = pd.read_csv(data.pred_path("ast", pipe, "test")).fillna("").set_index("seg_id").loc[A.seg_id].reset_index()
rows = []
for label, fn in [("BLEU", metrics.bleu_fn()), ("chrF++", metrics.chrf_fn())]:
    r = metrics.paired_bootstrap(fn, A.hyp.astype(str).tolist(), B.hyp.astype(str).tolist(), A.ref.astype(str).tolist(),
                                 n_samples=1000)
    rows.append({"métrique": label, "système A (direct)": direct, "système B (pipeline)": pipe,
                 "B - A": round(r["delta_moyen"], 2), "IC95 bas": round(r["IC95"][0], 2),
                 "IC95 haut": round(r["IC95"][1], 2), "p": round(r["p_value"], 3), "segments": len(A)})
out = pd.DataFrame(rows)
out.to_csv("results/ast_paired_test.csv", index=False)
print(out.to_string())
