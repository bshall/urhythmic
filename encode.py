import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.functional as AF

from urhythmic.model import encode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_dataset(args):
    logging.info("Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

    logging.info(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        if sr != 16000:
            raise ValueError(f"Sample rate: {sr} should be 16kHz.")
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units, log_probs = encode(hubert, wav)

        units_out_path = args.out_dir / "soft" / in_path.relative_to(args.in_dir)
        units_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(units_out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())

        probs_out_path = args.out_dir / "logprobs" / in_path.relative_to(args.in_dir)
        probs_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(probs_out_path.with_suffix(".npy"), log_probs.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset into soft speech units and the log probabilities of the associated discrete units."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .wav).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
