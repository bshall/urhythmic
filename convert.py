import argparse
import logging
from pathlib import Path

from tqdm import tqdm

import torch
import torchaudio
import torchaudio.functional as AF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPEAKERS = ["p228", "p268", "p225", "p232", "p257", "p231", "LJSpeech"]


def convert(args):
    logging.info("Loading HuBERT-Soft checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda()

    logging.info("Loading Urhythmic checkpoint")
    urhythmic, encode = torch.hub.load(
        "bshall/urhythmic:main",
        args.model,
        source_speaker=args.source,
        target_speaker=args.target,
        trust_repo=True,
    )
    urhythmic.cuda()

    logging.info(f"Coverting {args.in_dir} to {args.target}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = AF.resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units, log_probs = encode(hubert, wav)
            wav = urhythmic(units, log_probs)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            out_path.with_suffix(args.extension), wav.squeeze(0).cpu(), 16000
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio samples using Urhythmic."
    )
    parser.add_argument(
        "model",
        help="available models (Urhythmic-Fine or Urhythmic-Global).",
        choices=["urhythmic_fine", "urhythmic_global"],
    )
    parser.add_argument(
        "source",
        metavar="source-speaker",
        help=f"the source speaker: {', '.join(SPEAKERS)}",
        choices=SPEAKERS,
    )
    parser.add_argument(
        "target",
        metavar="target-speaker",
        help=f"the target speaker: {', '.join(SPEAKERS)}",
        choices=SPEAKERS,
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
    convert(args)
