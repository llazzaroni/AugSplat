# RadSplat

RadSplat is a small pipeline for:

1. preparing a COLMAP-style dataset with VGGT,
2. training a NeRF ensemble with Nerfstudio,
3. generating synthetic views from the ensemble,
4. exporting the same train/validation split to gsplat, and
5. training AugSplat variants on the augmented dataset.

The repository is organized as a normal public project:

- `radsplat/`: library code
- `scripts/`: runnable entrypoints
- `analysis/`: reporting and plotting utilities
- `submodules/`: vendored third-party dependencies

## Repository Layout

```text
RadSplat/
├── analysis/
│   ├── plot_gsplat_eval_stats.py
│   ├── rank_gsplat_combo.py
│   ├── report_best_nerf_ensemble.py
│   └── report_gsplat_reference_metrics.py
├── radsplat/
│   ├── augmentation/
│   ├── nerf/
│   └── splatting/
├── scripts/
│   ├── augment_dataset.py
│   ├── export_nerf_rays.py
│   ├── prepare_vggt_dataset.py
│   ├── run_scene_pipeline.sh
│   ├── run_splatting.py
│   └── sample_multiscale_dataset.py
└── submodules/
```

## Environments

The pipeline assumes three conda environments:

- `vggt`: VGGT reconstruction / depth export
- `nerfstudio`: Nerfstudio training and NeRF-side augmentation
- `gsplat`: gsplat / AugSplat training

The orchestration script switches between these environments automatically.

## Quick Start

The main entrypoint is:

```bash
bash scripts/run_scene_pipeline.sh --data-dir /path/to/scene_sparse
```

The input dataset must at least contain:

```text
scene_sparse/
└── images/
```

If `scene_sparse` is already a prepared VGGT dataset, the pipeline reuses it. Otherwise it creates:

- `scene_sparse_vggt`
- `scene_sparse_aug`
- `scene_sparse_artifacts`

### Default Behavior

By default the pipeline:

- prepares the VGGT dataset if needed,
- trains `5` `nerfacto` models for `15000` steps,
- saves/evaluates NeRF checkpoints every `200` steps,
- generates `200` synthetic views,
- exports the train/val split payload for gsplat,
- trains the staged AugSplat variant,
- saves the final gsplat checkpoint.

### Common Examples

Train the default staged AugSplat pipeline:

```bash
bash scripts/run_scene_pipeline.sh \
  --data-dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse \
  --camera-id 1
```

Train `depth-nerfacto` instead of `nerfacto`:

```bash
bash scripts/run_scene_pipeline.sh \
  --data-dir /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse \
  --nerf-method depth-nerfacto \
  --camera-id 1
```

Run the dual AugSplat variant:

```bash
bash scripts/run_scene_pipeline.sh \
  --data-dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse \
  --splat-mode dual \
  --dual-nerf-loss-weight 2 \
  --camera-id 1
```

Use a larger NeRF ensemble and a custom gsplat schedule:

```bash
bash scripts/run_scene_pipeline.sh \
  --data-dir /cluster/scratch/rbollati/new_exp/360_v2/treehill_sparse \
  --num-ensembles 8 \
  --nerf-max-steps 20000 \
  --splat-max-steps 15000 \
  --staged-nerf-phase-steps 500 \
  --camera-id 1
```

Delete intermediate artifacts after gsplat training and keep only the final gsplat result:

```bash
bash scripts/run_scene_pipeline.sh \
  --data-dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse \
  --cleanup-intermediate \
  --camera-id 1
```

## Pipeline Semantics

### 1. VGGT preparation

The script checks whether the input already looks like a prepared dataset:

- `images/`
- `images_2/`, `images_4/`, `images_8/`
- a valid COLMAP model under `sparse/` or `sparse/0/`
- `depths_vggt/` if `depth-nerfacto` is requested

If not, it runs:

```bash
python scripts/prepare_vggt_dataset.py \
  --src <input> \
  --dst <input>_vggt \
  --overwrite \
  --conf-thres-value 0.0
```

### 2. NeRF training

NeRF runs are stored in a stable artifact directory:

```text
<scene>_artifacts/nerf_models/
  nerf_ensemble_1/
  nerf_ensemble_2/
  ...
```

This is intentional. Nerfstudio outputs are sensitive to their run directory, so the repo does **not** store them inside `_aug` and move them later.

### 3. Synthetic view generation

The augmentation step uses:

```bash
python scripts/augment_dataset.py ...
```

By default the pipeline selects the **latest** NeRF checkpoint because that always works. You can switch to:

```bash
--checkpoint-selection best
```

if your Nerfstudio installation writes `eval_all_images.jsonl` for each run. You can also force a step with:

```bash
--checkpoint-step 2400
```

### 4. gsplat split payload

The repository still uses a `.pt` payload for gsplat split handoff. This is deliberate:

- gsplat already consumes `torch.load(...)`,
- the payload contains train/val filenames and camera metadata,
- no extra parser layer is needed.

The pipeline now uses:

```bash
python scripts/export_nerf_rays.py \
  --nerf-folder <config.yml> \
  --output-name <ray_sample.pt> \
  --split-only
```

so it exports only the split metadata needed by gsplat.

### 5. gsplat / AugSplat training

The public gsplat entrypoint is:

```bash
python scripts/run_splatting.py default ...
```

The pipeline exposes two modes:

- `--splat-mode staged`
- `--splat-mode dual`

`staged` in this repo means:

1. a NeRF-only warmup phase, then
2. a real-image gsplat phase

This is implemented with:

- `--staged-runner`

The meaningful staged knobs are:

- `--staged-nerf-phase-steps`
- `--staged-real-phase-steps`

## Manual Commands

The public scripts can also be called directly.

Prepare a dataset with VGGT:

```bash
python scripts/prepare_vggt_dataset.py \
  --src /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse \
  --dst /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse_vggt \
  --overwrite \
  --conf-thres-value 0.0
```

Generate an augmented dataset from an ensemble:

```bash
python scripts/augment_dataset.py \
  --model-roots /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse_artifacts/nerf_models \
  --checkpoint-selection latest \
  --input-dataset /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse_vggt \
  --output-dataset /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse_aug \
  --tmp-root /cluster/home/rbollati/tmp \
  --tau 1.5 \
  --camera-id 1 \
  --debug-plot-dir /cluster/scratch/rbollati/new_exp/360_v2/bicycle_sparse_artifacts/image_supervision \
  --num-final-samples 200 \
  --final-render-scale 0.125
```

Export the train/val split payload for gsplat:

```bash
python scripts/export_nerf_rays.py \
  --nerf-folder /path/to/config.yml \
  --output-name /path/to/ray_sample.pt \
  --split-only
```

Run staged AugSplat manually:

```bash
python scripts/run_splatting.py default \
  --no-nerf_init \
  --staged-runner \
  --data_dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_aug \
  --result_dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_artifacts/gsplat_staged \
  --pt_path /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_artifacts/ray_sample.pt \
  --data_factor 1 \
  --staged_nerf_phase_steps 300 \
  --staged_real_phase_steps 10000 \
  --nerf_samples_data_factor 8 \
  --batch_size 1 \
  --nerf_batch_factor 20 \
  --strategy.reset_every 100000000 \
  --deterministic \
  --disable_viewer \
  --save-last-ckpt
```

Run dual AugSplat manually:

```bash
python scripts/run_splatting.py default \
  --dual_runner \
  --no-nerf_init \
  --data_dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_aug \
  --result_dir /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_artifacts/gsplat_dual \
  --pt_path /cluster/scratch/rbollati/new_exp/360_v2/garden_sparse_artifacts/ray_sample.pt \
  --data_factor 1 \
  --nerf_samples_data_factor 8 \
  --batch_size 1 \
  --nerf_batch_factor 20 \
  --dual_nerf_loss_weight 2 \
  --dual_nerf_decay_steps_to_quarter 300 \
  --dual_nerf_disable_threshold 0.1 \
  --max_steps 10000 \
  --strategy.reset_every 100000000 \
  --deterministic \
  --disable_viewer
```

## Analysis

Compare gsplat runs:

```bash
python analysis/report_gsplat_reference_metrics.py \
  --run-dir /cluster/scratch/rbollati/new_exp/models_bicycle/gsplat \
  --run-dir /cluster/scratch/rbollati/new_exp/models_bicycle/gsplat_dual_1 \
  --label gsplat \
  --label dual_1 \
  --reference-run-dir /cluster/scratch/rbollati/new_exp/models_bicycle/gsplat \
  --source auto \
  --stage val \
  --step-interval 100
```

Plot gsplat statistics:

```bash
python analysis/plot_gsplat_eval_stats.py \
  --run-dir /cluster/scratch/rbollati/new_exp/models_garden/gsplat \
  --label 3DGS \
  --run-dir /cluster/scratch/rbollati/new_exp/models_garden/gsplat_dual_1 \
  --label "AugSplat dual" \
  --plot-style lines \
  --max-step 15000 \
  --out-dir /cluster/scratch/rbollati/new_exp/models_garden/gsplat_compare_plots
```

Report the best NeRF checkpoint in an ensemble:

```bash
python analysis/report_best_nerf_ensemble.py \
  --model-root /cluster/scratch/rbollati/new_exp/models_flowers
```

## Notes

- `--camera-id` is strongly recommended for multi-camera COLMAP datasets.
- If you want to use `--checkpoint-selection best`, make sure your Nerfstudio setup writes `eval_all_images.jsonl`.
- By default, the pipeline keeps all intermediate artifacts for debugging and analysis.
- `--camera-id 1` means “assign the synthetic views to COLMAP camera model id 1”, i.e. reuse that intrinsics entry for all generated `nerf_sample_*` images.
- `notes_1.txt` is kept as an internal lab notebook and is not part of the public API.
