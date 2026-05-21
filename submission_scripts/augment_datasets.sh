BASE_DIR="/cluster/scratch/rbollati/dataset"
MODELS_ROOT="/cluster/scratch/rbollati"
TMP_ROOT="/cluster/home/rbollati/tmp"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"

for d in "$BASE_DIR"/*_sparse_vggt; do
  [ -d "$d" ] || continue
  scene="$(basename "$d")"
  prefix="${scene%%_sparse_vggt}"

  echo "Scene: $scene"
  echo "Prefix: $prefix"

  model_dir="$MODELS_ROOT/models_${prefix}"

  for i in 1 2 3 4 5; do
    run_dir="$model_dir/nerf_ensemble_$i"
    [ -d "$run_dir" ] || { echo "Missing ensemble dir: $run_dir"; continue 2; }
  done

  cmd=(
    python ~/RadSplat/RadSplat/images.py
    --model-roots "$model_dir"
    --input-dataset "$BASE_DIR/$scene"
    --output-dataset "$BASE_DIR/${prefix}_sparse_aug"
    --tmp-root "$TMP_ROOT"
    --tau 1.5
    --debug-plot-dir image_supervision
    --num-final-samples 200
    --final-render-scale 0.125
    --camera-id 1
  )

  if [ -n "$CHECKPOINT_STEP" ]; then
    cmd+=(--checkpoint-step "$CHECKPOINT_STEP")
  fi

  "${cmd[@]}"
done
