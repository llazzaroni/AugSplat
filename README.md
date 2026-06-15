# RadSplat

Focused repo for the workflow used in this project:

1. prepare or downsample multiscale datasets
2. run VGGT to build `*_sparse_vggt`
3. train NeRF ensemble models
4. generate augmented datasets with synthetic `nerf_sample_*` views
5. export NeRF-aware train/val split metadata with `nerf_step.py`
6. train or evaluate Gaussian Splatting with dual or staged runners

The repo has been trimmed to keep only the code paths used by that workflow.

## Kept entrypoints

### Dataset preparation

- `utils/sample_multiscale_dataset.py`
- `utils/prepare_vggt_dataset.py`

Examples:

```bash
python utils/sample_multiscale_dataset.py \
  --src /cluster/scratch/.../scene \
  --dst /cluster/scratch/.../scene_subset \
  --n 30
```

```bash
python utils/prepare_vggt_dataset.py \
  --src /cluster/scratch/.../raw_scene \
  --dst /cluster/scratch/.../garden_sparse_vggt \
  --overwrite
```

### NeRF ensemble training

Train NeRF ensembles directly with `ns-train` from interactive shells.

### Augmented dataset generation

- `images.py`

Example:

```bash
python images.py \
  --model-roots /cluster/scratch/.../models_flowers \
  --checkpoint-step 2400 \
  --input-dataset /cluster/scratch/.../flowers_sparse_vggt \
  --output-dataset /cluster/scratch/.../flowers_sparse_aug \
  --tmp-root /cluster/home/.../tmp \
  --tau 1.5 \
  --num-final-samples 200 \
  --final-render-scale 0.125 \
  --camera-id 1
```

The output dataset is expected to contain:

- synthetic images under `images/`, `images_2/`, `images_4/`, `images_8/`
- weight maps under `weights_nerf_samples*`
- NeRF samples appended to `sparse/0/images.bin`

### NeRF ray export for gsplat

- `nerf_step.py`
- `nerfstep/`

Example:

```bash
python nerf_step.py \
  --nerf-folder /cluster/scratch/.../config.yml \
  --output-name /cluster/scratch/.../ray_sample.pt \
  --sampling-size 1000000 \
  --ray-sampling-strategy random
```

This produces the `.pt` payload used by gsplat to preserve train/val image splits.

### Gaussian splatting

- `experiments/run_gsplat.py`
- `gsplat_dir/`

Supported modes:

- `--dual_runner`
- `--staged-runner`
- both flags together for staged pretraining followed by dual training

Example:

```bash
python -m experiments.run_gsplat default \
  --no-nerf_init \
  --dual_runner \
  --staged-runner \
  --data_dir /cluster/scratch/.../garden_sparse_aug \
  --result_dir /cluster/scratch/.../models_garden/gsplat_staged_1 \
  --pt_path /cluster/scratch/.../models_garden/nerf_ensemble_1/outputs/ensemble_1/depth-nerfacto/garden/ray_sample.pt \
  --data_factor 1 \
  --staged_nerf_phase_steps 300 \
  --staged_real_phase_steps 15000 \
  --nerf_samples_data_factor 8 \
  --batch_size 1 \
  --nerf_batch_factor 20 \
  --dual_nerf_decay_steps_to_quarter 300 \
  --dual_nerf_disable_threshold 0.1 \
  --dual_nerf_loss_weight 0 \
  --max_steps 10000 \
  --strategy.reset_every 100000000 \
  --deterministic \
  --disable_viewer \
  --save-last-ckpt
```

Checkpoint behavior:

- default: save only `best_ckpt_rank*.pt`
- opt-in: add `--save-last-ckpt` to also write `last_ckpt_rank*.pt`

Checkpoint-only evaluation:

```bash
python -m experiments.run_gsplat default \
  --dual_runner \
  --no-nerf_init \
  --data_dir /cluster/scratch/.../garden_sparse_aug \
  --result_dir /cluster/scratch/.../models_garden/gsplat_eval \
  --pt_path /cluster/scratch/.../ray_sample.pt \
  --data_factor 1 \
  --nerf_samples_data_factor 8 \
  --disable_viewer \
  --disable_video \
  --ckpt /cluster/scratch/.../ckpts/best_ckpt_rank0.pt
```

This writes validation renders and, by default, training renders under:

- `result_dir/renders/val_*`
- `result_dir/renders/train_*`

## Plotting and reporting utilities

Kept helpers:

- `utils/plot_gsplat_eval_stats.py`
- `utils/report_gsplat_reference_metrics.py`
- `utils/rank_gsplat_combo.py`
- `utils/report_best_nerf_ensemble.py`

## Repo layout

```text
RadSplat/
  experiments/run_gsplat.py
  gsplat_dir/
  images.py
  nerf_step.py
  nerfstep/
  utils/
    plot_gsplat_eval_stats.py
    prepare_vggt_dataset.py
    rank_gsplat_combo.py
    report_best_nerf_ensemble.py
    report_gsplat_reference_metrics.py
    sample_multiscale_dataset.py
```

## Notes

- `notes_1.txt` was left untouched because it had local modifications outside this cleanup.
- `submodules/gsplat`, `submodules/nerfstudio`, and `submodules/vggt` are still required.
- `experiments/run_gsplat.py` is intentionally kept because the gsplat commands use `python -m experiments.run_gsplat ...`.
