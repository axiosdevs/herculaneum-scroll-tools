#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/pc/defi/vesuvius/.venv/bin/python
S3=https://vesuvius-challenge-open-data.s3.amazonaws.com
run () {
  NAME=$1; PRED=$2; CT=$3; CTLVL=$4
  [ -f "ct_support/survey_${NAME}.json" ] && { echo "skip $NAME"; return; }
  echo "=== SURVEY $NAME (ct-level $CTLVL) ==="
  $PY ct_support/ct_support.py survey --preds "$PRED" --ct "$CT" --ct-level "$CTLVL" \
    --stride 100 --out "ct_support/survey_${NAME}.json" 2>&1 | tail -2
}
run PHercMANBp "$S3/PHercMANBp/representations/predictions/surfaces/20251216152116-surface-20260413222639-surface-m7-L2-th0.2.zarr" "$S3/PHercMANBp/volumes/20251216152116-2.399um-0.2m-78keV-masked.zarr" 2
run PHerc1203 "$S3/PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr" "$S3/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" 2
run PHerc0257 "$S3/PHerc0257/representations/predictions/surfaces/20250821151750-surface-20260413222639-surface-m7-L0-th0.2.zarr" "$S3/PHerc0257/volumes/20250821151750-9.362um-1.2m-113keV-masked.zarr" 0
echo "CALIB_BATCH_COMPLETE"
