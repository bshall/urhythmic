from typing import List

import itertools

import torch
import torch.nn.functional as F

from urhythmic.utils import SoundType, SILENCE


class TimeStretcherFineGrained:
    """Time stretching block (Fine-Grained). Up/down samples the speech units to match the target rhythm."""

    def __call__(
        self,
        units: torch.Tensor,
        clusters: List[SoundType],
        boundaries: List[int],
        tgt_duartations: List[int],
    ) -> torch.Tensor:
        """
        Args:
            units (Tensor): soft speech units of shape (1, D, T)
                where D is the dimension of the units and T is the number of frames.
            clusters (List[SoundType]): list of sound types for each segment of shape (N,)
                where N is the number of segments.
            boundaries (List[int]): list of segment bounaries of shape (N+1,).
            tgt_durations (List[int]): list of target durations of shape (N,).
        Returns:
            Tensor: up/down sampled soft speech units.
        """
        units = [
            units[..., t0:tn]
            for cluster, (t0, tn) in zip(clusters, itertools.pairwise(boundaries))
            if cluster not in SILENCE or tn - t0 > 3
        ]
        units = [
            F.interpolate(segment, mode="linear", size=duration)
            for segment, duration in zip(units, tgt_duartations)
        ]
        units = torch.cat(units, dim=-1)
        return units


class TimeStretcherGlobal:
    """Time stretching block (Global). Up/down samples the speech units to match the target speaking rate."""

    def __call__(self, units: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Args:
            units (Tensor): soft speech units of shape (1, D, T)
                where D is the dimension of the units and T is the number of frames.
            ratio (float): ratio between the source and target speaking rates.
        Returns:
            Tensor: up/down sampled soft speech units.
        """
        units = F.interpolate(units, scale_factor=ratio, mode="linear")
        return units
