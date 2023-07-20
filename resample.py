import argparse
import logging
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torchaudio
import torchaudio.functional as AF
import numpy as np
import itertools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resample_file(path, sample_rate):
    wav, sr = torchaudio.load(path)
    wav = AF.resample(wav, sr, sample_rate)
    torchaudio.save(path, wav, sample_rate)
    return wav.size(-1) / sample_rate


def resample_dataset(args):
    logger.info(f"Resampling dataset at {args.in_dir}")
    paths = list(args.in_dir.rglob("*.wav"))
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(
            tqdm(
                executor.map(resample_file, paths, itertools.repeat(args.sample_rate)),
                total=len(paths),
            )
        )
    logger.info(f"Processed {np.sum(results) / 60 / 60:4f} hours of audio.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to dataset directory.",
    )
    parser.add_argument(
        "--sample_rate",
        help="target sample rate (defaults to 16000).",
        type=int,
        default=16000,
    )
    args = parser.parse_args()
    resample_dataset(args)
