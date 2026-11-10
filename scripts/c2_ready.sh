#!/usr/bin/env bash
# Autorise le départ de C2 dans scripts/run_02_e2e2_c2.sh : tests unitaires sur CPU, code commité, puis écriture de
# /root/runs/C2_READY = HEAD du projet. Le script de nuit n'attend que ce fichier : à relancer après tout nouveau
# commit (même de résultats seuls) fait avant le départ de C2, sinon C2 continue d'attendre.
#
#   wsl -d Ubuntu-22.04 --cd "<projet>" -- bash scripts/c2_ready.sh
source /root/tedxtn-env.sh > /dev/null 2>&1
G() { git -c safe.directory="*" -C "$PROJ" "$@"; }
H=$(G rev-parse HEAD)
if [ -n "$(G status --porcelain common notebooks tests scripts)" ]; then
  echo "REFUS : modifications non commitées dans common/ notebooks/ tests/ scripts/"; G status --short common notebooks tests scripts; exit 1
fi
CUDA_VISIBLE_DEVICES= python tests/test_core.py 2>&1 | grep -vi warn; t=${PIPESTATUS[0]}
[ "$t" = 0 ] || { echo "REFUS : tests en échec"; exit 1; }
echo "$H" > /root/runs/C2_READY && echo "C2_READY = $H ($(G log -1 --format=%s))"
