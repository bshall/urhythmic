import argparse
import logging
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch
import numpy as np
import itertools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def segment_file(segmenter, in_path, out_path):
    log_probs = np.load(in_path)
    segments, boundaries = segmenter(log_probs)
    np.savez(out_path.with_suffix(".npz"), segments=segments, boundaries=boundaries)
    return log_probs.shape[0], np.mean(np.diff(boundaries))


def segment_dataset(args):
    logging.info("Loading segmenter checkpoint")
    segmenter = torch.hub.load("bshall/urhythmic:main", "segmenter", num_clusters=3)

    in_paths = list(args.in_dir.rglob("*.npy"))
    out_paths = [args.out_dir / path.relative_to(args.in_dir) for path in in_paths]

    logger.info("Setting up folder structure")
    for path in tqdm(out_paths):
        path.parent.mkdir(exist_ok=True, parents=True)

    logger.info("Segmenting dataset")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(
                    segment_file,
                    itertools.repeat(segmenter),
                    in_paths,
                    out_paths,
                ),
                total=len(in_paths),
            )
        )

    frames, boundary_length = zip(*results)
    logger.info(f"Segmented {sum(frames) * 0.02 / 60 / 60} hours of audio")
    logger.info(f"Average segment length: {np.mean(boundary_length) * 0.02} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to the log probability directory.",
    )
    parser.add_argument(
        "out_dir", metavar="out-dir", type=Path, help="path to the output directory."
    )
    args = parser.parse_args()
    segment_dataset(args)
