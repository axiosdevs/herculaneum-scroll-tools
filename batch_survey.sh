#!/bin/bash
# Batch surveys for the 2026-04-13 m7 prediction batch (villa#1114)
cd "$(dirname "$0")"
PY=/Users/pc/defi/vesuvius/.venv/bin/python
S3=https://vesuvius-challenge-open-data.s3.amazonaws.com

run () {
  NAME=$1; PRED=$2; CT=$3; CTLVL=$4
  [ -f "ct_support/survey_${NAME}.json" ] && { echo "skip $NAME (done)"; return; }
  echo "=== SURVEY $NAME (ct-level $CTLVL) ==="
  $PY ct_support/ct_support.py survey --preds "$PRED" --ct "$CT" --ct-level "$CTLVL" \
    --stride 100 --out "ct_support/survey_${NAME}.json" 2>&1 | tail -3
}

run PHerc1451 "$S3/PHerc1451/representations/predictions/surfaces/20260319101107-surface-20260413222639-surface-m7-L2-th0.2.zarr" \
             "$S3/PHerc1451/volumes/20260319101107-2.399um-0.2m-78keV-masked.zarr" 2

run PHerc1299 "$S3/PHerc1299/representations/predictions/surfaces/20260309130042-surface-20260413222639-surface-m7-L2-th0.2.zarr" \
             "$S3/PHerc1299/volumes/20260309130042-2.399um-0.2m-78keV-masked.zarr" 2

run PHerc0814_24um "$S3/PHerc0814/representations/predictions/surfaces/20260309142202-surface-20260413222639-surface-m7-L2-th0.2.zarr" \
             "$S3/PHerc0814/volumes/20260309142202-2.399um-0.2m-78keV-masked.zarr" 2

run PHerc0814_94um "$S3/PHerc0814/representations/predictions/surfaces/20250804134230-surface-20260413222639-surface-m7-L0-th0.2.zarr" \
             "$S3/PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr" 0

echo "BATCH_SURVEYS_COMPLETE"
