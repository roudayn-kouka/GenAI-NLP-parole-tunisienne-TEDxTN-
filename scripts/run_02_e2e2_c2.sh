#!/usr/bin/env bash
# Notebook 02 en local (WSL2, GPU 6 Go), sans surveillance : E2E-2 (Whisper-small translate initialisé depuis
# asr_whisper-small/final), puis C2 (NLLB LoRA) quand le code qu'il utilise est validé.
#
#   bash scripts/run_02_e2e2_c2.sh <commit_e2e2>     # se détache ; journal : /root/logs/run_02_e2e2_c2.log
#
# 0. attend la fin de scripts/run_01_whisper.sh s'il tourne encore, puis supprime
#    models/asr_whisper-small/checkpoint-3500 (non retenu : le modèle final est le meilleur sur la dev loss,
#    checkpoint-2750 ; suppression seulement si final/ existe et que c'est bien le cas) ;
# 1. E2E-2 depuis une copie figée du code (git archive <commit_e2e2> dans /root/runs/code_<commit>) : les
#    modifications faites pendant la nuit dans le projet (common/, notebooks/) ne le touchent pas ; prédictions et
#    journaux écrits dans results/ du projet (TEDXTN_RESULTS_DIR) ;
# 2. C2 depuis le projet, seulement quand /root/runs/C2_READY contient le commit HEAD du projet (fichier écrit
#    une fois les corrections commitées et les tests passés) : sinon le script attend.
# Autres sections du notebook 02 coupées ; COMET coupé (passe finale). Prédictions existantes relues.
set -u
cd "$(dirname "$0")/.."
LOG=/root/logs/run_02_e2e2_c2.log
if [ "${1:-}" != "--fg" ]; then
  [ -n "${1:-}" ] || { echo "usage : bash $0 <commit_e2e2>"; exit 1; }
  mkdir -p /root/logs /root/runs
  cp "$0" /root/runs/run_02_e2e2_c2.sh   # bash lit le script au fil de l'eau : copie, le projet peut changer cette nuit
  setsid nohup bash /root/runs/run_02_e2e2_c2.sh --fg "$1" > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 3
  echo "lancé en arrière-plan (PID $!) ; suivi : tail -f $LOG"
  exit 0
fi
COMMIT_E2E2=$2
source /root/tedxtn-env.sh
GIT="git -c safe.directory=* -C $PROJ"

if pgrep -f "run_01_whisper.sh --fg" > /dev/null; then
  echo "=== attente de la fin de run_01_whisper.sh $(date '+%F %T')"
  while pgrep -f "run_01_whisper.sh --fg" > /dev/null; do sleep 60; done
fi
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
export TEDXTN_RESULTS_DIR="$PROJ/results"
pgrep -f gpu_monitor.sh > /dev/null || { setsid nohup bash scripts/gpu_monitor.sh /root/logs/gpu_monitor.csv 300 87 \
  > /dev/null 2>&1 < /dev/null & }

run() {  # run <dossier du code> <sortie>
  echo "=== $2 début $(date '+%F %T') (code : $1)"
  (cd "$1" && jupyter nbconvert --to notebook --execute notebooks/02_Speech_to_Text_Translation.ipynb --output-dir /root/runs \
    --output "$2" --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 2>&1 \
    | grep -vE "IPKernelApp|Deprecated Feature|MissingIDFieldWarning|_validate\(" | tail -5)
  echo "=== $2 fin $(date '+%F %T')"
}

# 1. E2E-2 (code figé)
CODE=/root/runs/code_$COMMIT_E2E2
rm -rf "$CODE" && mkdir -p "$CODE" && $GIT archive "$COMMIT_E2E2" | tar -x -C "$CODE" || { echo "git archive impossible"; exit 1; }
TEDXTN_RUN_NLLB_FT=0 TEDXTN_RUN_E2E=1 TEDXTN_RUN_E2E1=0 TEDXTN_RUN_E2E2=1 run "$CODE" 02_e2e2

# 2. C2 (code du projet, une fois validé)
echo "=== C2 : attente de /root/runs/C2_READY = HEAD du projet $(date '+%F %T')"
while true; do
  head=$($GIT rev-parse HEAD 2>/dev/null)
  [ -f /root/runs/C2_READY ] && [ "$(cat /root/runs/C2_READY)" = "$head" ] && break
  sleep 120
done
echo "code validé : $head"
TEDXTN_RUN_NLLB_FT=1 TEDXTN_RUN_E2E=0 run "$PROJ" 02_c2
ls -la "$PROJ/results/preds/" | grep -E "ast__" || echo "aucune prédiction ast"
echo "alertes température (> 87 °C soutenu) : $(grep -c ALERTE /root/logs/gpu_monitor.csv)"
pkill -f gpu_monitor.sh
echo "FIN $(date '+%F %T')"
