#!/usr/bin/env bash
# Notebook 02 en local (WSL2, GPU 6 Go), sans surveillance : C2 (NLLB LoRA) puis E2E-2 (Whisper-small translate
# initialisé depuis asr_whisper-small/final).
#
#   bash scripts/run_02_c2_e2e2.sh          # se détache et rend la main ; journal : /root/logs/run_02_c2_e2e2.log
#
# 0. attend la fin de scripts/run_01_whisper.sh (ligne « FIN » de son journal) s'il tourne encore, puis supprime
#    models/asr_whisper-small/checkpoint-3500 (dernier checkpoint après l'early stopping, non retenu : le modèle final
#    est le meilleur sur la dev loss, checkpoint-2750 ; suppression seulement si final/ existe et que c'est bien le cas) ;
# 1. passe C2 : fine-tuning NLLB-600M (LoRA) puis cascade sur les prédictions ASR dev/test du système retenu sur le
#    dev par le notebook 01 (models/asr_best/SOURCE.txt) et oracle (transcriptions de référence) sur le test ;
# 2. passe E2E-2 : entraînement puis prédictions dev/test (beam 4).
# Les autres sections du notebook 02 sont désactivées ; COMET coupé (passe finale). Prédictions existantes relues.
set -u
cd "$(dirname "$0")/.."
LOG=/root/logs/run_02_c2_e2e2.log
if [ "${1:-}" != "--fg" ]; then
  mkdir -p /root/logs /root/runs
  setsid nohup bash "$0" --fg > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 3
  echo "lancé en arrière-plan (PID $!) ; suivi : tail -f $LOG"
  exit 0
fi

source /root/tedxtn-env.sh
L01=/root/logs/run_01_whisper.log
if pgrep -f "run_01_whisper.sh --fg" > /dev/null; then
  echo "=== attente de la fin de run_01_whisper.sh $(date '+%F %T')"
  while pgrep -f "run_01_whisper.sh --fg" > /dev/null; do sleep 60; done
fi
grep -q "^FIN " "$L01" && echo "run_01 terminé : $(grep '^FIN ' "$L01")" || echo "ATTENTION : run_01 sans ligne FIN (voir $L01)"

M="$TEDXTN_WORKDIR/models/asr_whisper-small"
if [ -d "$M/checkpoint-3500" ] && [ -f "$M/final/config.json" ]; then
  best=$(python -c "import json; print(json.load(open('$M/checkpoint-3500/trainer_state.json'))['best_model_checkpoint'].rsplit('/',1)[1])")
  if [ "$best" = "checkpoint-2750" ]; then
    rm -rf "$M/checkpoint-3500" && echo "checkpoint-3500 supprimé (final = $best, choisi sur la dev loss)"
  else
    echo "checkpoint-3500 GARDÉ : best_model_checkpoint = $best (attendu checkpoint-2750)"
  fi
fi

export TEDXTN_SKIP_DOWNLOAD=1 TEDXTN_MAX_BATCH=4 TEDXTN_OPTIM=adamw_bnb_8bit TEDXTN_REUSE_PREDS=1
export TEDXTN_RUN_ZERO_SHOT=0 TEDXTN_RUN_SEAMLESS=0 TEDXTN_RUN_CASCADE=0 TEDXTN_RUN_LLM=0 TEDXTN_RUN_COMET=0
export TEDXTN_RUN_OTHER_TARGETS=0 TEDXTN_RUN_ASR_NOISE=0 TEDXTN_RUN_ENC_DEC_DIAGNOSTIC=0
pgrep -f gpu_monitor.sh > /dev/null || { setsid nohup bash scripts/gpu_monitor.sh /root/logs/gpu_monitor.csv 300 87 \
  > /dev/null 2>&1 < /dev/null & }
echo "ASR source de la cascade : $(cat "$TEDXTN_WORKDIR/models/asr_best/SOURCE.txt" 2>/dev/null)"

run() {  # run <sortie>
  echo "=== $1 début $(date '+%F %T')"
  jupyter nbconvert --to notebook --execute notebooks/02_Speech_to_Text_Translation.ipynb --output-dir /root/runs \
    --output "$1" --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 2>&1 \
    | grep -vE "IPKernelApp|Deprecated Feature|MissingIDFieldWarning|_validate\(" | tail -5
  echo "=== $1 fin $(date '+%F %T')"
}
TEDXTN_RUN_NLLB_FT=1 TEDXTN_RUN_E2E=0 run 02_c2
TEDXTN_RUN_NLLB_FT=0 TEDXTN_RUN_E2E=1 TEDXTN_RUN_E2E1=0 TEDXTN_RUN_E2E2=1 run 02_e2e2
ls -la "$PROJ/results/preds/" | grep -E "ast__" || echo "aucune prédiction ast"
echo "alertes température (> 87 °C soutenu) : $(grep -c ALERTE /root/logs/gpu_monitor.csv)"
pkill -f gpu_monitor.sh
echo "FIN $(date '+%F %T')"
