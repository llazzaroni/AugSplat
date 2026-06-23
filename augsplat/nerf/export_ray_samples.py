import warnings
import math
import torch
import logging

from augsplat.nerf.models import Nerfacto
from augsplat.nerf.point_samplers import (
    sobel_edge_detector_sampler,
    canny_edge_detector_sampler,
    random_sampler,
    mixed_sampler,
    patched_sampler,
)
from augsplat.nerf.gs_initializer import Initializer
from pathlib import Path
import argparse

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

def create_parser():
    parser = argparse.ArgumentParser(
        description="Export NeRF-derived initialization samples and/or the train/val split payload used by gsplat."
    )

    parser.add_argument(
        "--nerf-folder",
        "-nf",
        type=str,
        required=True,
        help="Path to the Nerfstudio config.yml used to load the NeRF model."
    )

    parser.add_argument(
        "--output-name",
        "-o",
        type=str,
        required=True,
        help="Output .pt payload path."
    )

    parser.add_argument(
        "--sampling-size",
        "-n",
        type=int,
        default=0,
        help="Number of rays to sample. Required unless --split-only is set."
    )

    parser.add_argument(
        "--ray-sampling-strategy",
        "-s",
        type=str,
        required=False,
        help="Sampling strategy: canny | sobel | mixed-sobel | mixed-canny | patches | random"
    )

    parser.add_argument(
        "--percentage-random",
        "-pr",
        type=float,
        required=False,
        help="Random-ray share for mixed samplers."
    )

    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only export train/val split metadata for gsplat without sampling NeRF rays.",
    )

    return parser

def rel_to_images_root(p: str) -> str:
    # find 'images' in the path and take from there (COLMAP parser returns names relative to images/)
    parts = Path(p).parts
    if "images" in parts:
        i = parts.index("images")
        return str(Path(*parts[i:]))  # e.g. 'images/seq/frame0001.png'
    return Path(p).name

def main():
    parser = create_parser()
    args = parser.parse_args()

    folder = args.nerf_folder
    N_RAYS = args.sampling_size
    BATCH_SIZE = 5_000
    RAYS_BATCH_NAME = args.output_name
    n_batches = math.ceil(N_RAYS / BATCH_SIZE)

    
    print('########### loading model')
    model = Nerfacto(folder)
    cams = model.pipeline.datamanager.train_dataset.cameras.to('cpu')
    dpo = model.pipeline.datamanager.train_dataparser_outputs
    test_split = getattr(model.pipeline.datamanager, "test_split", "test")
    test_dpo = model.pipeline.datamanager.dataparser.get_dataparser_outputs(split=test_split)
    xyzrgb = None

    if not args.split_only:
        if N_RAYS <= 0:
            raise ValueError("--sampling-size must be > 0 unless --split-only is set.")

        if args.ray_sampling_strategy == "canny":
            coords = canny_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
        elif args.ray_sampling_strategy == "sobel":
            coords = sobel_edge_detector_sampler(model.pipeline.datamanager, N_RAYS, model.device)
        elif args.ray_sampling_strategy == "mixed-sobel":
            coords = mixed_sampler(model.pipeline.datamanager, N_RAYS, share_rnd = args.percentage_random, edge_detector = "sobel", device = model.device)
        elif args.ray_sampling_strategy == "mixed-canny":
            coords = mixed_sampler(model.pipeline.datamanager, N_RAYS, share_rnd = args.percentage_random, edge_detector = "canny", device = model.device)
        elif args.ray_sampling_strategy == "patches":
            coords = patched_sampler(model.pipeline.datamanager, N_RAYS, model.device, 32, 16)
        else:
            coords = random_sampler(model.pipeline.datamanager, N_RAYS, model.device)

        xyzrgb_chunks = []
        for b in range(n_batches):

            # get initial and final index of the index rays to query
            s = b * BATCH_SIZE
            e = min((b + 1) * BATCH_SIZE, N_RAYS)
            if s >= e:
                break
            batch_rays_indexes = coords[s:e, :]

            logging.info('sampling rays')
            rays = model.create_rays(batch_rays_indexes)

            logging.info('sampling points')

            sampled = model.sample_points(rays)
            outputs, field_outputs, weights  = model.evaluate_points(sampled)

            logging.info('init gs')
            gs_initializer = Initializer(weights, sampled)

            logging.info('compute trasmittance')
            trasmittance = gs_initializer.compute_transmittance()

            logging.info('computer initial_position')
            initial_position = gs_initializer.compute_inital_positions(trasmittance, 0.5)

            xyzrgb_batch = torch.cat([initial_position, outputs['rgb']], dim=-1)

            xyzrgb_chunks.append(xyzrgb_batch.detach().cpu())


        xyzrgb = torch.cat(xyzrgb_chunks, dim=0)
        print("number of rays:", xyzrgb.shape[0])

    image_filenames_abs = [str(p) for p in dpo.image_filenames]
    image_filenames_rel = [rel_to_images_root(p) for p in image_filenames_abs]

    val_image_filenames_abs = [str(p) for p in test_dpo.image_filenames]
    val_image_filenames_rel = [rel_to_images_root(p) for p in val_image_filenames_abs]

    # Per-image split metadata (keyed by relative image path).
    split_by_image_rel = {name: "train" for name in image_filenames_rel}
    for name in val_image_filenames_rel:
        # In case of overlap, train split takes precedence.
        split_by_image_rel.setdefault(name, test_split)

    payload = {
        "camera_to_worlds": cams.camera_to_worlds.cpu(),
        "K": cams.get_intrinsics_matrices().cpu(),
        "image_filenames_abs": image_filenames_abs,
        "image_filenames_rel": image_filenames_rel,
        "train_image_filenames_abs": image_filenames_abs,
        "train_image_filenames_rel": image_filenames_rel,
        "val_split_name": test_split,
        "val_image_filenames_abs": val_image_filenames_abs,
        "val_image_filenames_rel": val_image_filenames_rel,
        "split_by_image_rel": split_by_image_rel,
    }
    if xyzrgb is not None:
        payload["xyzrgb"] = xyzrgb.cpu()

    logging.info('saving initial positions')
    torch.save(payload, RAYS_BATCH_NAME)


if __name__ == "__main__":
    main()
