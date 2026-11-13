#!/usr/bin/env bash
# Lance l'interface (app/app.py) dans WSL avec les modèles finaux (asr_best, ast_best, mt_best), en arrière-plan.
# Les trois modèles sont chargés AVANT l'ouverture du serveur, et le journal dit lequel a été chargé (dossier,
# SOURCE.txt) ou si un modèle de base de repli a été pris à sa place.
#
#   wsl -d Ubuntu-22.04 --cd "<projet>" -- bash scripts/run_app.sh          # puis http://localhost:7860 sous Windows
#   wsl -d Ubuntu-22.04 --cd "<projet>" -- bash scripts/run_app.sh stop     # arrête l'interface
set -u
LOG=/root/logs/app.log
PORT=${TEDXTN_APP_PORT:-7860}
if [ "${1:-}" = "stop" ]; then
  pkill -f "tedxtn_app_server" && echo "interface arrêtée" || echo "aucune interface en cours"; exit 0
fi
if [ "${1:-}" != "--fg" ]; then
  mkdir -p /root/logs /root/runs
  pgrep -f "tedxtn_app_server" > /dev/null && { echo "interface déjà lancée : http://localhost:$PORT (journal : $LOG)"; exit 0; }
  cp "$0" /root/runs/run_app.sh
  setsid nohup bash /root/runs/run_app.sh --fg > "$LOG" 2>&1 < /dev/null &
  disown
  echo "lancement en arrière-plan ; chargement des modèles (≈ 1 min) ; journal : $LOG"
  for i in $(seq 1 90); do
    sleep 2
    grep -q "Running on local URL" "$LOG" && break
    grep -qE "Traceback|REPLI" "$LOG" && break
  done
  grep -E "^\[modèle\]|REPLI|Traceback|Error|Running on" "$LOG"
  grep -q "Running on local URL" "$LOG" && echo "-> ouvrir sous Windows : http://localhost:$PORT"
  exit 0
fi
source /root/tedxtn-env.sh
exec python - tedxtn_app_server "$PORT" <<'PY'   # « tedxtn_app_server » : repère pour pgrep / stop
import sys
from pathlib import Path
sys.path.insert(0, ".")
from common.config import AppModels
from common.pipeline import TunisianSpeechPipeline
from app.app import launch

port = int(sys.argv[2])
models = AppModels()
pipe = TunisianSpeechPipeline(models)
for role, path, backend in [("ASR", models.asr, lambda: pipe.asr), ("traduction directe", models.ast_e2e, lambda: pipe.e2e),
                            ("NLLB (pipeline)", models.mt, lambda: pipe.mt)]:
    b = backend()   # chargement maintenant, pas au premier clic
    src = Path(path) / "SOURCE.txt"
    origin = src.read_text().strip() if src.exists() else "?"
    if b.warning:
        print(f"[modèle] {role} : REPLI sur un modèle de base -> {b.warning}", flush=True)
    else:
        print(f"[modèle] {role} : {path} (source : {origin})", flush=True)
launch(share=False, port=port, pipeline=pipe, server_name="0.0.0.0")
PY
