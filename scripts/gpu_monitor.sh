#!/usr/bin/env bash
# Surveille le GPU pendant les entraînements longs : une ligne toutes les INTERVAL secondes (défaut 300 = 5 min).
#   bash scripts/gpu_monitor.sh [fichier_log] [intervalle_s] [seuil_°C]
# Écrit aussi une ligne « ALERTE » quand la température dépasse le seuil (défaut 87 °C) sur 3 mesures consécutives
# (≥ 10 min de suite avec l'intervalle par défaut) : chauffe prolongée, à signaler.
LOG=${1:-$HOME/logs/gpu_monitor.csv}
INTERVAL=${2:-300}
LIMIT=${3:-87}
mkdir -p "$(dirname "$LOG")"
[ -s "$LOG" ] || echo "date,temperature.gpu,clocks.sm [MHz],utilization.gpu [%],memory.used [MiB],power.draw [W]" > "$LOG"
hot=0
while true; do
  line=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date '+%F %T'),${line// /}" >> "$LOG"
  t=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  if [[ "$t" =~ ^[0-9]+$ ]] && [ "$t" -gt "$LIMIT" ]; then hot=$((hot + 1)); else hot=0; fi
  if [ "$hot" -ge 3 ]; then echo "$(date '+%F %T'),ALERTE temperature ${t} C > ${LIMIT} C depuis ${hot} mesures" >> "$LOG"; fi
  sleep "$INTERVAL"
done
