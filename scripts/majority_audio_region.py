# Classe majoritaire sur le même échantillon audio (40 segments par talk) et les mêmes plis GroupKFold que la
# cellule 15 du notebook 04 (région par l'audio). Exécute les cellules 1 à 9 du notebook (données, régions ; pas de
# GPU, aucun entraînement). Usage : python scripts/majority_audio_region.py "$(pwd)"
import json, os, sys
os.chdir(sys.argv[1]); sys.path.insert(0, os.getcwd())

nb = json.load(open("notebooks/04_Dialect_Identification.ipynb", encoding="utf-8"))
ns = {"display": lambda *a, **k: None}
for i in range(1, 10):
    c = nb["cells"][i]
    if c["cell_type"] == "code":
        exec("".join(c["source"]).replace("display(", "(lambda *a, **k: None)("), ns)
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
data, df, keep, SEED = ns["data"], ns["df"], ns["keep"], ns["SEED"]
Dall = data.cut_segments(df)
Da = Dall[Dall.region.isin(keep)].groupby("talk", group_keys=False).apply(lambda x: x.sample(min(40, len(x)), random_state=SEED)).reset_index(drop=True)
y, g = Da.region.values, Da.talk.values
n_folds = min(5, Da.groupby("region").talk.nunique().min())
print("échantillon audio :", len(Da), "segments,", Da.talk.nunique(), "talks ; plis :", n_folds)
print("répartition :", Da.region.value_counts().to_dict())
glob = pd.Series(y).value_counts().index[0]
pred_glob = np.array([glob] * len(y))
pred_fold = np.empty(len(y), dtype=object)
for tr, te in GroupKFold(n_folds).split(np.zeros(len(y)), y, g):
    pred_fold[te] = pd.Series(y[tr]).value_counts().index[0]
for name, p in [("classe majoritaire (globale)", pred_glob), ("classe majoritaire du pli d'apprentissage", pred_fold)]:
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    print(f"{name} : accuracy {accuracy_score(y, p):.3f}, F1 macro {f1:.3f}")
