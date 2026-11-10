#!/usr/bin/env bash
# État des exécutions locales (WSL2) : processus, checkpoints, courbes du DEV, prédictions présentes, GPU.
# Aucune métrique du test n'est affichée.
#
#   wsl -d Ubuntu-22.04 --cd "<projet>" -- bash scripts/status.sh
source /root/tedxtn-env.sh > /dev/null 2>&1
echo "=== $(date '+%F %T') | WSL démarré : $(uptime -s)"
echo "--- exécutions en cours"
pgrep -af "run_0[12]|run_local_jobs|jupyter-nbconvert" | grep -v pgrep | cut -c1-150 || echo "aucune"
echo "--- fin des journaux"
for l in /root/logs/run_*.log; do [ -f "$l" ] && { echo "[$(basename "$l")]"; grep -E "^===|^FIN|checkpoint-|Traceback|Error" "$l" | tail -4; }; done
echo "--- modèles"
python - <<'PY'
import glob, json, os
M = os.environ["TEDXTN_WORKDIR"] + "/models"
for d in sorted(glob.glob(M + "/*/")):
    name = os.path.basename(d.rstrip("/"))
    ck = sorted(glob.glob(d + "checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    fin = os.path.exists(d + "final/config.json")
    line = f"{name:34s} final={'oui' if fin else 'non'}"
    if ck:
        s = json.load(open(ck[-1] + "/trainer_state.json")); h = s["log_history"]
        ev = [(x["step"], round(x["eval_loss"], 4)) for x in h if "eval_loss" in x][-3:]
        dv = [(x["step"], round(x.get("dev_WER_subset", x.get("dev_BLEU_subset", float("nan"))), 2))
              for x in h if "dev_WER_subset" in x or "dev_BLEU_subset" in x][-3:]
        best = (s.get("best_model_checkpoint") or "").rsplit("/", 1)[-1]
        pat = s.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {}).get("attributes", {}).get("early_stopping_patience_counter")
        line += (f" | step {s['global_step']}/{s['max_steps']} | dev loss {ev} | dev WER/BLEU (200 seg.) {dv}"
                 f" | meilleur {best} | patience {pat}/3")
    print(line)
PY
echo "--- prédictions (noms seulement)"
ls "$PROJ/results/preds/" | grep -v gitkeep
echo "--- GPU"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
echo "alertes > 87 °C : $(grep -c ALERTE /root/logs/gpu_monitor.csv 2>/dev/null) | dernier relevé : $(tail -1 /root/logs/gpu_monitor.csv 2>/dev/null)"
echo "C2_READY : $(cat /root/runs/C2_READY 2>/dev/null || echo absent) | HEAD : $(git -c safe.directory='*' -C "$PROJ" rev-parse HEAD)"
