#!/bin/bash
# Bring a fresh rented box to a running survey in one command.
#
# Everything here exists because of a day lost to the alternatives: processes started over ssh
# do not survive the session, `pkill -f name.py` matches the very shell doing the killing, and
# a running worker keeps the old code in memory after its file is patched. So: launchers live
# in files, patterns are written 'na[m]e.py', and every stage is restarted by its own script.
#
#   ./bootstrap.sh            # meshes from the local backup, then the survey
#   STAGE=survey ./bootstrap.sh
set -e
WS=${WS:-/workspace}
VENV=${VENV:-/venv/main/bin/python}
VOL=${VOL:-https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr/0/}
NSH=${NSH:-6}            # GPU renderers; each holds 2-3 GB of card, so 6 fits 24 GB beside inference
THR=${THR:-32}           # fetch threads per renderer -- the card is never the bottleneck, S3 is
POLARITY=${POLARITY:-fwd}
STAGE=${STAGE:-all}

cd "$WS"

need() { [ -f "$1" ] || { echo "нет $1 — скопируйте из репозитория/архива"; exit 1; }; }

if [ "$STAGE" = "all" ] || [ "$STAGE" = "check" ]; then
  for f in render_gpu.py rw_whole.py infer_whole.py run_r152.py r152.ckpt; do need "$f"; done
  [ -d ink ] || { echo "нет ink/ — скопируйте модули"; exit 1; }
  $VENV - <<'PY'
import torch
print("карта:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "НЕТ CUDA")
PY
  echo "проверка пройдена"
fi

stop_all() {
  pkill -f 'rw_who[l]e.py' 2>/dev/null || true
  pkill -f 'infer_who[l]e.py' 2>/dev/null || true
  pkill -f 'scan_seati[n]g.py' 2>/dev/null || true
  sleep 4
  pkill -9 -f 'render_g[p]u.py' 2>/dev/null || true
  pkill -9 -f 'render_t[r]i.py' 2>/dev/null || true
  sleep 2
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "seating" ]; then
  stop_all
  rm -f "$WS"/seating_scan_*.json
  for i in $(seq 0 19); do
    VOLURL="$VOL" SHARD=$i NSHARD=20 THREADS=8 \
      setsid $VENV -u "$WS/scan_seating.py" > "$WS/ss_$i.log" 2>&1 < /dev/null &
  done
  echo "ранжировщиков: $(pgrep -fc 'scan_seati[n]g.py') — около минуты на поверхность"
  while pgrep -f 'scan_seati[n]g.py' > /dev/null; do sleep 30; done
  $VENV "$WS/pick_best.py"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "survey" ]; then
  stop_all
  PICKSF=${PICKSF:-$WS/picks_flat.json}
  need "$PICKSF"
  CKPT=$WS/r152.ckpt BATCH=4 POLARITY=$POLARITY \
    setsid $VENV -u "$WS/infer_whole.py" > "$WS/inf.log" 2>&1 < /dev/null &
  sleep 8
  for i in $(seq 0 $((NSH-1))); do
    RENDERER=render_gpu.py VOLURL="$VOL" PICKS="$PICKSF" SHARD=$i NSHARD=$NSH \
      THREADS=$THR MAXQ=2 \
      setsid $VENV -u "$WS/rw_whole.py" > "$WS/rd_$i.log" 2>&1 < /dev/null &
  done
  sleep 5
  echo "рендеров $(pgrep -fc 'rw_who[l]e.py'), инференс $(pgrep -fc 'infer_who[l]e.py'), направление $POLARITY"
  echo "следить:  grep -h ПОЛОТНО $WS/inf.log | tail"
  echo "простой карты:  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
fi
