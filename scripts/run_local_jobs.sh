#!/usr/bin/env bash
# Tâches GPU locales (WSL2, GPU 6 Go), enchaînées sans surveillance, une à la fois.
#
#   bash scripts/run_local_jobs.sh c01 llmcorr c4 audio c3   # se détache ; journal : /root/logs/run_local_jobs.log
#
# Tâches (dans l'ordre donné ; l'asr_best est whisper-small-ft) :
#   c01     : notebook 02, C0 (oracle) et C1 (ASR retenu) avec NLLB zero-shot ;
#   llmcorr : notebook 01 §10, post-correction de l'ASR par LLM (Qwen2.5-1.5B, exemples few-shot du train ; aucun
#             réentraînement, prédictions ASR relues) ;
#   c4      : notebook 02, C4 (traduction par LLM) sur la transcription corrigée ; sautée si llmcorr n'a pas produit
#             la transcription corrigée du dev et du test (C4 en dépend) ;
#   audio   : notebook 04, embeddings w2v-BERT-2.0 (identification de région / code-switching par l'audio) ;
#   c3      : notebook 02, C3 (NLLB LoRA entraîné avec 50 % de sorties ASR bruitées du train, inférence ASR sur le
#             train comprise) ; C2 relu ;
#   byt5    : notebook 03, ByT5-small (normalisation seq2seq), 1 epoch (≈ 1 250 steps) ; LLM, MSA et impact MT coupés ;
#   e2e1    : notebook 02, E2E-1 (Whisper-small translate depuis le modèle de base) ; autres sections coupées.
# Passe finale (aucun réentraînement : modèles finals trouvés -> entraînement sauté ; prédictions relues) :
#   final01 : notebook 01 complet (évaluation dev + test) ;
#   final02 : notebook 02 complet (B0 zero-shot, autres langues cibles, COMET, test apparié direct / pipeline) ;
#             E2E-1 coupé (abandonné) ; inférence ASR du train de C3 coupée (les prédictions de C3 sont relues) ;
#             B0 = Whisper-small zero-shot seulement (large-v3 : plus d'1 h 30 sans finir sur 6 Go, chiffre du papier) ;
#   final03 : notebook 03 complet en un passage : ByT5 1 epoch (contrôle de recopie) + LLM, MSA, impact MT ;
#   final04 : notebook 04 complet ;  final05 : notebook 05 (démo ; limite de 30 min, l'interface Gradio est lancée) ;
#   platform: scripts/platform_tests.py (20 segments du test x 3 modes x 3 langues -> results/platform_tests.csv).
# Attend d'abord la fin de toute exécution de notebook en cours. Une tâche en échec est signalée (« ÉCHEC ») et la
# suivante est lancée. Chaque notebook exécuté va dans /root/runs/<tâche>.ipynb ; le notebook du projet n'est pas
# modifié. Code : celui du projet au lancement de chaque tâche.
set -u
LOG=/root/logs/run_local_jobs.log
if [ "${1:-}" != "--fg" ]; then
  [ $# -ge 1 ] || { echo "usage : bash $0 c01|llmcorr|c4|audio|c3|byt5|e2e1 ..."; exit 1; }
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
while busy; do   # un script peut attendre entre deux notebooks : on attend le script, pas seulement le notebook
  echo "=== attente : une autre exécution GPU tourne encore ($(date '+%F %T'))"; sleep 120
done

export TEDXTN_SKIP_DOWNLOAD=1 TEDXTN_MAX_BATCH=4 TEDXTN_OPTIM=adamw_bnb_8bit TEDXTN_REUSE_PREDS=1
# notebook 02 : tout coupé, chaque tâche rallume ce qu'il lui faut (la dernière valeur d'une variable l'emporte)
OFF02="TEDXTN_RUN_ZERO_SHOT=0 TEDXTN_RUN_SEAMLESS=0 TEDXTN_RUN_CASCADE=0 TEDXTN_RUN_LLM=0 TEDXTN_RUN_COMET=0 TEDXTN_RUN_OTHER_TARGETS=0 TEDXTN_RUN_ENC_DEC_DIAGNOSTIC=0 TEDXTN_RUN_E2E=0 TEDXTN_RUN_NLLB_FT=0 TEDXTN_RUN_ASR_NOISE=0"
OFF01="TEDXTN_RUN_BASELINES=0 TEDXTN_RUN_WHISPER_FT=0 TEDXTN_RUN_CTC_FT=0 TEDXTN_RUN_MMS=0 TEDXTN_RUN_AUG_ABLATION=0"
FAILED=""

run() {  # run <notebook> <sortie> <variables...>
  local nb=$1 out=$2; shift 2
  pgrep -f gpu_monitor.sh > /dev/null || { setsid nohup bash scripts/gpu_monitor.sh /root/logs/gpu_monitor.csv 300 87 \
    > /dev/null 2>&1 < /dev/null & }   # relancé à chaque tâche
  local to=-1; [ "$out" = 05_final ] && to=1800   # le notebook 05 lance l'interface : jamais bloquant plus de 30 min
  echo "=== $out début $(date '+%F %T') ($*)"
  env "$@" jupyter nbconvert --to notebook --execute "notebooks/$nb" --output-dir /root/runs --output "$out" \
    --ExecutePreprocessor.timeout=$to --ExecutePreprocessor.kernel_name=python3 2>&1 \
    | grep -vE "IPKernelApp|Deprecated Feature|MissingIDFieldWarning|_validate\(" | tail -25
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then echo "=== $out fin $(date '+%F %T')"
  else echo "=== $out ÉCHEC (code $rc) $(date '+%F %T') : voir les lignes ci-dessus et /root/runs/$out.ipynb"; FAILED="$FAILED $out"; fi
}
corrected() { ls "$PROJ"/results/preds/ | grep -q "^asr__$(cat "$TEDXTN_WORKDIR/models/asr_best/SOURCE.txt")+llm__$1.csv$"; }

for job in "$@"; do
  case $job in
    c01)     run 02_Speech_to_Text_Translation.ipynb 02_c01 $OFF02 TEDXTN_RUN_CASCADE=1 ;;
    llmcorr) run 01_Tunisian_ASR.ipynb 01_llmcorr $OFF01 TEDXTN_RUN_LLM_CORRECTION=1 ;;
    c4)      if corrected dev && corrected test; then run 02_Speech_to_Text_Translation.ipynb 02_c4 $OFF02 TEDXTN_RUN_LLM=1
             else echo "=== 02_c4 SAUTÉE $(date '+%F %T') : transcription corrigée par LLM absente (llmcorr)"; FAILED="$FAILED 02_c4(sautée)"; fi ;;
    audio)   run 04_Dialect_Identification.ipynb 04_audio TEDXTN_RUN_AUDIO=1 ;;
    c3)      run 02_Speech_to_Text_Translation.ipynb 02_c3 $OFF02 TEDXTN_RUN_NLLB_FT=1 TEDXTN_RUN_ASR_NOISE=1 ;;
    byt5)    run 03_Text_Normalization.ipynb 03_byt5 TEDXTN_RUN_BYT5=1 TEDXTN_BYT5_EPOCHS=1 TEDXTN_RUN_LLM=0 TEDXTN_RUN_MSA=0 TEDXTN_RUN_MT_IMPACT=0 ;;
    e2e1)    run 02_Speech_to_Text_Translation.ipynb 02_e2e1 $OFF02 TEDXTN_RUN_E2E=1 TEDXTN_RUN_E2E1=1 TEDXTN_RUN_E2E2=0 ;;
    final01) run 01_Tunisian_ASR.ipynb 01_final ;;
    final02) run 02_Speech_to_Text_Translation.ipynb 02_final TEDXTN_RUN_E2E1=0 TEDXTN_RUN_ASR_NOISE=0 TEDXTN_B0_MODELS=whisper-small ;;
    final03) run 03_Text_Normalization.ipynb 03_final TEDXTN_RUN_BYT5=1 TEDXTN_BYT5_EPOCHS=1 ;;   # ByT5 + tout le notebook
    final04) run 04_Dialect_Identification.ipynb 04_final ;;
    final05) run 05_Interface_Demo.ipynb 05_final ;;
    platform)
      echo "=== platform début $(date '+%F %T')"
      python scripts/platform_tests.py 2>&1 | grep -viE "warn" | tail -15
      if [ "${PIPESTATUS[0]}" = 0 ]; then echo "=== platform fin $(date '+%F %T')"
      else echo "=== platform ÉCHEC $(date '+%F %T')"; FAILED="$FAILED platform"; fi ;;
    *)       echo "tâche inconnue : $job" ;;
  esac
done
echo "tâches en échec ou sautées :${FAILED:- aucune}"
echo "alertes température (> 87 °C soutenu) : $(grep -c ALERTE /root/logs/gpu_monitor.csv)"
pkill -f gpu_monitor.sh
echo "FIN $(date '+%F %T')"
