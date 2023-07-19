from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as transforms


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
    ):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="slaney",
        )
        self.pad = (win_length - hop_length) // 2

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = F.pad(wav, (self.pad, self.pad), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


class MelDataset(Dataset):
    def __init__(
        self,
        root: Path,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
    ):
        split = "train" if train else "dev"
        self.wavs_dir = root / split / "wavs"
        self.units_dir = root / split / "soft"

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train

        self.metadata = [
            path.relative_to(self.wavs_dir).with_suffix("")
            for path in self.wavs_dir.rglob("*.wav")
        ]

        self.logmel = LogMelSpectrogram()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.metadata[index]
        wav_path = self.wavs_dir / path
        units_path = self.units_dir / path

        info = torchaudio.info(wav_path.with_suffix(".wav"))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        units = torch.from_numpy(np.load(units_path.with_suffix(".npy"))).float()

        units_frames_per_segment = math.floor(self.segment_length / self.hop_length)
        units_diff = units.size(0) - units_frames_per_segment if self.train else 0
        units_offset = random.randint(0, max(units_diff, 0))

        frame_offset = self.hop_length * units_offset

        wav, _ = torchaudio.load(
            filepath=wav_path.with_suffix(".wav"),
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if self.train:
            units = units[units_offset : units_offset + units_frames_per_segment, :]

        if units.size(0) < units_frames_per_segment:
            diff = units_frames_per_segment - units.size(0)
            units = F.pad(units, (0, diff), "constant", units.mean())

        return wav, units.transpose(0, 1), tgt_logmel
