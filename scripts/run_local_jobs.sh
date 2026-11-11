#!/usr/bin/env bash
# Tâches GPU locales (WSL2, GPU 6 Go), enchaînées sans surveillance, une à la fois.
#
#   bash scripts/run_local_jobs.sh audio byt5 e2e1     # se détache ; journal : /root/logs/run_local_jobs.log
#
# Tâches (dans l'ordre donné) :
#   audio : notebook 04, embeddings w2v-BERT-2.0 (identification de région / code-switching par l'audio) ;
#   byt5  : notebook 03, ByT5-small (normalisation seq2seq), 1 epoch (≈ 1 250 steps) ; LLM, MSA et impact MT coupés (passe finale) ;
#   e2e1  : notebook 02, E2E-1 (Whisper-small translate depuis le modèle de base) ; autres sections coupées ;
#   c3    : notebook 02, C3 (NLLB LoRA entraîné sur des sorties ASR bruitées du train) ; C2 relu.
# Attend d'abord la fin de toute exécution de notebook en cours (nuit précédente). Chaque notebook exécuté va dans
# /root/runs/<tâche>.ipynb ; le notebook du projet n'est pas modifié. Code : celui du projet au lancement.
set -u
LOG=/root/logs/run_local_jobs.log
if [ "${1:-}" != "--fg" ]; then
  [ $# -ge 1 ] || { echo "usage : bash $0 audio|byt5|e2e1|c3 ..."; exit 1; }
  mkdir -p /root/logs /root/runs
  cp "$0" /root/runs/run_local_jobs.sh   # bash lit le script au fil de l'eau : copie
  setsid nohup bash /root/runs/run_local_jobs.sh --fg "$@" > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 3
  echo "lancé en arrière-plan (PID $!) : $* ; suivi : tail -f $LOG"
  exit 0
fi
shift
source /root/tedxtn-env.sh
busy() { pgrep -f "jupyter-nbconvert|run_01_whisper.sh --fg|run_02_e2e2_c2.sh --fg" > /dev/null; }
while busy; do   # le script de la nuit peut attendre entre deux notebooks : on attend le script, pas le notebook
  echo "=== attente : une autre exécution GPU tourne encore ($(date '+%F %T'))"; sleep 120
done

export TEDXTN_SKIP_DOWNLOAD=1 TEDXTN_MAX_BATCH=4 TEDXTN_OPTIM=adamw_bnb_8bit TEDXTN_REUSE_PREDS=1
OFF02="TEDXTN_RUN_ZERO_SHOT=0 TEDXTN_RUN_SEAMLESS=0 TEDXTN_RUN_CASCADE=0 TEDXTN_RUN_LLM=0 TEDXTN_RUN_COMET=0 TEDXTN_RUN_OTHER_TARGETS=0 TEDXTN_RUN_ENC_DEC_DIAGNOSTIC=0"

run() {  # run <notebook> <sortie> <variables...>
  local nb=$1 out=$2; shift 2
  pgrep -f gpu_monitor.sh > /dev/null || { setsid nohup bash scripts/gpu_monitor.sh /root/logs/gpu_monitor.csv 300 87 \
    > /dev/null 2>&1 < /dev/null & }   # relancé à chaque tâche (le script de la nuit l'arrête en finissant)
  echo "=== $out début $(date '+%F %T') ($*)"
  env "$@" jupyter nbconvert --to notebook --execute "notebooks/$nb" --output-dir /root/runs --output "$out" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 2>&1 \
    | grep -vE "IPKernelApp|Deprecated Feature|MissingIDFieldWarning|_validate\(" | tail -5
  echo "=== $out fin $(date '+%F %T')"
}
for job in "$@"; do
  case $job in
    audio) run 04_Dialect_Identification.ipynb 04_audio TEDXTN_RUN_AUDIO=1 ;;
    byt5)  run 03_Text_Normalization.ipynb 03_byt5 TEDXTN_RUN_BYT5=1 TEDXTN_BYT5_EPOCHS=1 TEDXTN_RUN_LLM=0 TEDXTN_RUN_MSA=0 TEDXTN_RUN_MT_IMPACT=0 ;;
    e2e1)  run 02_Speech_to_Text_Translation.ipynb 02_e2e1 $OFF02 TEDXTN_RUN_NLLB_FT=0 TEDXTN_RUN_ASR_NOISE=0 \
             TEDXTN_RUN_E2E=1 TEDXTN_RUN_E2E1=1 TEDXTN_RUN_E2E2=0 ;;
    c3)    run 02_Speech_to_Text_Translation.ipynb 02_c3 $OFF02 TEDXTN_RUN_E2E=0 TEDXTN_RUN_NLLB_FT=1 TEDXTN_RUN_ASR_NOISE=1 ;;
    *)     echo "tâche inconnue : $job" ;;
  esac
done
echo "alertes température (> 87 °C soutenu) : $(grep -c ALERTE /root/logs/gpu_monitor.csv)"
pkill -f gpu_monitor.sh
echo "FIN $(date '+%F %T')"
