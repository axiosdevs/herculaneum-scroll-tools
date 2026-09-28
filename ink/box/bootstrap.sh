#!/bin/bash
# Reproduce the PHerc1451 survey on a fresh rented GPU box, from this repository alone.
#
#   git clone https://github.com/axiosdevs/herculaneum-scroll-tools && cd herculaneum-scroll-tools
#   ./ink/box/bootstrap.sh                 # stage, rank all 320 surfaces, survey the flat ones
#   STAGE=survey PICKS=all ./ink/box/bootstrap.sh   # every surface, not only the flat 43
#
# The survey scripts expect the layout they ran in, /workspace, so that is where this stages
# them. Everything they need ships here: the 320 grown meshes (ink/p1451/meshes, 28 MB), the
# pick lists, the renderers and the inference wrapper. The checkpoint is fetched once.
#
# Three things a day was lost to, and why this is written the way it is: processes started
# over ssh do not outlive the session, so every stage is detached with setsid; `pkill -f
# name.py` matches the shell doing the killing, so patterns are written 'na[m]e.py'; and a
# running worker keeps its old code in memory after its file is patched, so every stage
# restarts its own workers.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WS=/workspace
VENV=${VENV:-python3}
VOL=${VOL:-https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr/0/}
CKPT_URL=https://huggingface.co/scrollprize/ink_canonical_2um/resolve/main/r152_3ddec_v2_l5_epoch13.ckpt
NSH=${NSH:-6}            # GPU renderers; each holds 2-3 GB of card, six fit 24 GB beside inference
THR=${THR:-32}           # fetch threads per renderer: S3, not the card, is the bottleneck
POLARITY=${POLARITY:-fwd}   # measured: reverse is the blind face -- see ink/detectability.py
STAGE=${STAGE:-all}
PICKS=${PICKS:-flat}

stop_all() {
  pkill -f 'rw_who[l]e.py' 2>/dev/null || true
  pkill -f 'infer_who[l]e.py' 2>/dev/null || true
  pkill -f 'scan_seati[n]g.py' 2>/dev/null || true
  sleep 4
  pkill -9 -f 'render_g[p]u.py' 2>/dev/null || true
  pkill -9 -f 'render_t[r]i.py' 2>/dev/null || true
  sleep 2
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage" ]; then
  mkdir -p "$WS/ink"
  cp "$REPO"/ink/*.py "$WS/ink/"
  cp "$REPO"/ink/survey/*.py "$WS/"
  cp "$REPO/ink/render_gpu.py" "$WS/"
  cp -R "$REPO"/ink/p1451/meshes/n1451_r* "$WS/"
  cp "$REPO/ink/p1451/picks1451.json" "$REPO/ink/p1451/picks_flat.json" "$WS/"
  [ -f "$WS/r152.ckpt" ] || { echo "качаю чекпойнт (1.4 ГБ, один раз)"; curl -sL -o "$WS/r152.ckpt" "$CKPT_URL"; }
  $VENV - <<'PY'
import torch
print("карта:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "НЕТ CUDA")
PY
  echo "разложено в $WS: $(ls -d $WS/n1451_r* | wc -l) папок сеток"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "seating" ]; then
  stop_all
  rm -f "$WS"/seating_scan_*.json
  cd "$WS"
  for i in $(seq 0 19); do
    VOLURL="$VOL" PICKS="$WS/picks1451.json" SHARD=$i NSHARD=20 THREADS=8 \
      setsid $VENV -u "$WS/scan_seating.py" > "$WS/ss_$i.log" 2>&1 < /dev/null &
  done
  echo "ранжирую посадку: около полутора минут на поверхность, двадцать потоков"
  while pgrep -f 'scan_seati[n]g.py' > /dev/null; do sleep 30; done
  $VENV "$WS/pick_best.py"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "survey" ]; then
  stop_all
  cd "$WS"
  if [ "$PICKS" = "all" ]; then PICKSF=$WS/picks1451.json; else PICKSF=$WS/picks_flat.json; fi
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
  echo "полотна:        grep -h ПОЛОТНО $WS/inf.log | tail"
  echo "лист полотен:   $VENV $WS/contact.py"
  echo "простой карты:  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
fi
