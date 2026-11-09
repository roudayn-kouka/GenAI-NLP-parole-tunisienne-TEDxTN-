#!/usr/bin/env bash
# Notebook 01 en local (WSL2, GPU 6 Go) : Whisper-small fine-tuné puis baselines zero-shot, sans surveillance.
#
#   bash scripts/run_01_whisper.sh          # se détache (setsid/nohup) et rend la main ; journal : /root/logs/run_01_whisper.log
#
# Enchaînement (chaque étape reprend ou saute ce qui existe déjà) :
#   1. scripts/recompute_dev_wer.py : points WER dev perdus lors d'un arrêt (checkpoints encore présents, dev seul) ;
#   2. notebook 01, entraînement : reprise depuis le checkpoint-* le plus récent de models/asr_whisper-small
#      (run_whisper_ft), puis inférence dev + test (glouton) et test beam 5 ; si final/ existe, l'entraînement est
#      sauté ; prédictions déjà présentes relues (TEDXTN_REUSE_PREDS=1) ;
#   3. notebook 01, baselines zero-shot (whisper-small, whisper-large-v3) sur le test.
# Les notebooks exécutés vont dans /root/runs (le notebook du projet n'est pas modifié). Température GPU : gpu_monitor.csv.
set -u
cd "$(dirname "$0")/.."
LOG=/root/logs/run_01_whisper.log
if [ "${1:-}" != "--fg" ]; then
  mkdir -p /root/logs /root/runs
  setsid nohup bash "$0" --fg > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 3
  echo "lancé en arrière-plan (PID $!) ; suivi : tail -f $LOG"
  exit 0
fi

source /root/tedxtn-env.sh
export TEDXTN_SKIP_DOWNLOAD=1 TEDXTN_MAX_BATCH=4 TEDXTN_OPTIM=adamw_bnb_8bit TEDXTN_REUSE_PREDS=1
export TEDXTN_RUN_CTC_FT=0 TEDXTN_RUN_LLM_CORRECTION=0 TEDXTN_RUN_MMS=0
pgrep -f gpu_monitor.sh > /dev/null || { setsid nohup bash scripts/gpu_monitor.sh /root/logs/gpu_monitor.csv 300 87 \
  > /dev/null 2>&1 < /dev/null & }

run() {  # run <sortie>
  echo "=== $1 début $(date '+%F %T')"
  jupyter nbconvert --to notebook --execute notebooks/01_Tunisian_ASR.ipynb --output-dir /root/runs --output "$1" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 2>&1 | grep -vE "IPKernelApp|Deprecated Feature" | tail -5
  echo "=== $1 fin $(date '+%F %T')"
}
echo "=== recalcul WER dev $(date '+%F %T')"
python scripts/recompute_dev_wer.py 2>&1 | grep -viE "warn|deprecat"
echo "checkpoints : $(ls -d "$TEDXTN_WORKDIR"/models/asr_whisper-small/checkpoint-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
TEDXTN_RUN_BASELINES=0 TEDXTN_RUN_WHISPER_FT=1 run 01_train
TEDXTN_RUN_BASELINES=1 TEDXTN_RUN_WHISPER_FT=0 run 01_baselines
ls -la "$PROJ/results/preds/"
grep -c ALERTE /root/logs/gpu_monitor.csv | sed 's/^/alertes température (> 87 °C soutenu) : /'
pkill -f gpu_monitor.sh
echo "FIN $(date '+%F %T')"
