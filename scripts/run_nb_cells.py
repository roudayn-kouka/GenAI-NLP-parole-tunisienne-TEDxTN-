"""Exécute seulement certaines cellules de code d'un notebook (dans l'ordre donné, un seul espace de noms), sans
toucher au notebook. Sert à refaire une mesure isolée sans relancer le reste (CHANGELOG n° 61).

    python scripts/run_nb_cells.py notebooks/02_Speech_to_Text_Translation.ipynb 3 4 5 7 12 18 39

Les sections coûteuses doivent être coupées par les interrupteurs TEDXTN_RUN_<NOM>=0 ; `display` affiche le texte.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)


def _display(*objs, **_):
    for o in objs:
        print(o.to_string() if hasattr(o, "to_string") else o)


def main(path, cells):
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for i in cells:
        c = nb["cells"][i]
        if c["cell_type"] != "code":
            raise SystemExit(f"la cellule {i} n'est pas une cellule de code")
        src = "".join(c["source"])
        print(f"===== cellule {i}", flush=True)
        exec(compile(src, f"{path}#cellule{i}", "exec"), ns)
        ns["display"] = _display   # après les imports du notebook, qui importent display d'IPython


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
