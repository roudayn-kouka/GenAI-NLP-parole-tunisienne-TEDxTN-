#!/usr/bin/env bash
# Construit un modèle de langage 4-gram KenLM sur les transcriptions du TRAIN uniquement
# (jamais dev/test, pour éviter toute fuite). Utilisé par le décodeur CTC (w2v-BERT-2.0).
#
# Usage :  bash scripts/build_kenlm.sh <corpus_train.txt> <dossier_sortie> [ordre=4]
# Le fichier texte est produit par le notebook 01 (une phrase normalisée CTC par ligne).
set -euo pipefail
CORPUS=${1:?fichier texte du train}
OUT=${2:?dossier de sortie}
ORDER=${3:-4}
mkdir -p "$OUT"

if ! command -v lmplz >/dev/null 2>&1; then
  echo ">> Compilation de KenLM (≈ 3 min sur Colab)"
  if command -v apt-get >/dev/null 2>&1 && ! { command -v cmake && dpkg -s libboost-program-options-dev; } >/dev/null 2>&1; then
    # « apt-get update » échoue dès qu'un dépôt tiers est mal signé (vu sur une machine de l'équipe) : ce n'est pas
    # bloquant si les paquets restent installables, donc on n'arrête pas le script sur cette erreur.
    apt-get -qq update >/dev/null 2>&1 || echo ">> apt-get update : erreurs partielles (dépôt tiers), on continue"
    apt-get -qq install -y build-essential cmake libboost-system-dev \
      libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev \
      libbz2-dev liblzma-dev >/dev/null
  fi
  BUILD=${KENLM_BUILD_DIR:-/tmp/kenlm}
  [ -d "$BUILD" ] || git clone -q https://github.com/kpu/kenlm "$BUILD"
  mkdir -p "$BUILD/build" && cd "$BUILD/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j"$(nproc)" lmplz build_binary >/dev/null
  cd - >/dev/null
  export PATH="$BUILD/build/bin:$PATH"
fi

echo ">> Estimation du ${ORDER}-gram"
lmplz -o "$ORDER" --discount_fallback < "$CORPUS" > "$OUT/tedxtn_${ORDER}gram.arpa"

# pyctcdecode a besoin de la ligne </s> dans l'ARPA (correctif classique de la recette HF)
python3 - "$OUT/tedxtn_${ORDER}gram.arpa" <<'PY'
import sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().splitlines()
if not any("\t</s>" in l for l in lines):
    out, added = [], False
    for l in lines:
        if not added and "<s>" in l and l.split("\t")[1:2] == ["<s>"]:
            out.append(l)
            out.append(l.replace("<s>", "</s>"))
            added = True
            continue
        if l.startswith("ngram 1="):
            n = int(l.split("=")[1]); l = f"ngram 1={n + 1}"
        out.append(l)
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY

build_binary "$OUT/tedxtn_${ORDER}gram.arpa" "$OUT/tedxtn_${ORDER}gram.bin"
echo ">> OK : $OUT/tedxtn_${ORDER}gram.bin"
