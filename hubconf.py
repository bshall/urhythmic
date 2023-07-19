dependencies = ["torch", "torchaudio", "numpy", "scipy", "numba", "sklearn"]

URLS = {
    "segmenter-3": "https://github.com/bshall/urhythmic/releases/download/v0.1/segmenter-3-61beaeac.pt",
    "segmenter-8": "https://github.com/bshall/urhythmic/releases/download/v0.1/segmenter-8-b3d14f93.pt",
    "rhythm-model-fine-grained": "https://github.com/bshall/urhythmic/releases/download/v0.1/rhythm-fine-143621e1.pt",
    "rhythm-model-global": "https://github.com/bshall/urhythmic/releases/download/v0.1/rhythm-global-745d52d8.pt",
    "hifigan-p228": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p228-4ab1748f.pt",
    "hifigan-p268": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p268-36a1d51a.pt",
    "hifigan-p225": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p225-cc447edc.pt",
    "hifigan-p232": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p232-e0efc4c3.pt",
    "hifigan-p257": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p257-06fd495b.pt",
    "hifigan-p231": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-p231-250198a1.pt",
    "hifigan-LJSpeech": "https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-LJSpeech-ceb1368d.pt",
}

SPEAKERS = {"p228", "p268", "p225", "p232", "p257", "p231", "LJSpeech"}

from typing import Tuple, Callable

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from urhythmic.model import UrhythmicFine, UrhythmicGlobal, encode
from urhythmic.segmenter import Segmenter
from urhythmic.rhythm import RhythmModelFineGrained, RhythmModelGlobal
from urhythmic.stretcher import TimeStretcherFineGrained, TimeStretcherGlobal
from urhythmic.vocoder import HifiganGenerator, HifiganDiscriminator


def segmenter(
    num_clusters: int,
    gamma: float = 2,
    pretrained: bool = True,
    progress=True,
) -> Segmenter:
    """Segmentation and clustering block. Groups similar speech units into short segments.
    The segments are then combined into coarser groups approximating sonorants, obstruents, and silences.

    Args:
        num_clusters (int): number of clusters used for agglomerative clustering.
        gamma (float): regularizer weight encouraging longer segments
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.

    Returns:
        Segmenter: the segmentation and clustering block (optionally pretrained).
    """
    segmenter = Segmenter(num_clusters=num_clusters, gamma=gamma)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[f"segmenter-{num_clusters}"],
            progress=progress,
        )
        segmenter.load_state_dict(checkpoint)
    return segmenter


def rhythm_model_fine_grained(
    source_speaker: None | str,
    target_speaker: None | str,
    pretrained: bool = True,
    progress=True,
) -> RhythmModelFineGrained:
    """Rhythm modeling block (Fine-Grained). Estimates the duration distribution of each sound type.

    Available speakers:
        VCTK: p228, p268, p225, p232, p257, p231.
        LJSpeech.

    Args:
        source_speaker (None | str): the source speaker. None to fit your own source speaker or a selection from the available speakers.
        target_speaker (None | str): the target speaker. None to fit your own source speaker or a selection from the available speakers.
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.

    Returns:
        RhythmModelFineGrained: the fine-grained rhythm modeling block (optionally preloaded with source and target duration models).
    """
    if source_speaker is not None and source_speaker not in SPEAKERS:
        raise ValueError(f"source speaker is not in available set: {SPEAKERS}")
    if target_speaker is not None and target_speaker not in SPEAKERS:
        raise ValueError(f"target speaker is not in available set: {SPEAKERS}")

    rhythm_model = RhythmModelFineGrained()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["rhythm-model-fine-grained"],
            progress=progress,
        )
        state_dict = {}
        if target_speaker:
            state_dict["target"] = checkpoint[target_speaker]
        if source_speaker:
            state_dict["source"] = checkpoint[source_speaker]
        rhythm_model.load_state_dict(state_dict)
    return rhythm_model


def rhythm_model_global(
    source_speaker: None | str,
    target_speaker: None | str,
    pretrained: bool = True,
    progress=True,
) -> RhythmModelGlobal:
    """Rhythm modeling block (Global). Estimates speaking rate.

    Available speakers:
        VCTK: p228, p268, p225, p232, p257, p231.
        LJSpeech.

    Args:
        source_speaker (None | str): the source speaker. None to fit your own source speaker or a selection from the available speakers.
        target_speaker (None | str): the target speaker. None to fit your own source speaker or a selection from the available speakers.
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.

    Returns:
        RhythmModelGlobal: the global rhythm modeling block (optionally preloaded with source and target speaking rates).
    """
    if source_speaker is not None and source_speaker not in SPEAKERS:
        raise ValueError(f"source speaker is not in available set: {SPEAKERS}")
    if target_speaker is not None and target_speaker not in SPEAKERS:
        raise ValueError(f"target speaker is not in available set: {SPEAKERS}")

    rhythm_model = RhythmModelGlobal()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["rhythm-model-global"],
            progress=progress,
        )
        state_dict = {}
        if target_speaker:
            state_dict["target_rate"] = checkpoint[target_speaker]
        if source_speaker:
            state_dict["source_rate"] = checkpoint[source_speaker]
        rhythm_model.load_state_dict(state_dict)
    return rhythm_model


def hifigan_generator(
    speaker: None | str,
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> HifiganGenerator:
    """HifiGAN Generator.

    Available speakers:
        VCTK: p228, p268, p225, p232, p257, p231.
        LJSpeech.

    Args:
        speaker (None | str): the target speaker. None to fit your own speaker or a selection from the available speakers.
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.
        map_location: function or a dict specifying how to remap storage locations (see torch.load)

    Returns:
        HifiganGenerator: the HifiGAN Generator (pretrained on LJSpeech or one of the VCTK speakers).
    """
    if speaker is not None and speaker not in SPEAKERS:
        raise ValueError(f"target speaker is not in available set: {SPEAKERS}")

    hifigan = HifiganGenerator()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[f"hifigan-{speaker}"], map_location=map_location, progress=progress
        )
        consume_prefix_in_state_dict_if_present(
            checkpoint["generator"]["model"], "module."
        )
        hifigan.load_state_dict(checkpoint["generator"]["model"])
        hifigan.eval()
        hifigan.remove_weight_norm()
    return hifigan


def hifigan_discriminator(
    pretrained: bool = True, progress: bool = True, map_location=None
) -> HifiganDiscriminator:
    """HifiGAN Discriminator.

    Args:
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.
        map_location: function or a dict specifying how to remap storage locations (see torch.load)

    Returns:
        HifiganDiscriminator: the HifiGAN Discriminator (pretrained on LJSpeech).
    """
    discriminator = HifiganDiscriminator()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["hifigan-LJSpeech"], map_location=map_location, progress=progress
        )
        consume_prefix_in_state_dict_if_present(
            checkpoint["discriminator"]["model"], "module."
        )
        discriminator.load_state_dict(checkpoint["discriminator"]["model"])
        discriminator.eval()
    return discriminator


def urhythmic_fine(
    source_speaker: str | None,
    target_speaker: str | None,
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> Tuple[UrhythmicFine, Callable]:
    """Urhythmic (Fine-Grained), a voice and rhythm conversion system that does not require text or parallel data.

    Available speakers:
        VCTK: p228, p268, p225, p232, p257, p231.
        LJSpeech.

    Args:
        source_speaker (None | str): the source speaker. None to fit your own source speaker or a selection from the available speakers.
        target_speaker (None | str): the target speaker. None to fit your own source speaker or a selection from the available speakers.
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.
        map_location: function or a dict specifying how to remap storage locations (see torch.load)

    Returns:
        UrhythmicFine: the Fine-Grained Urhythmic model.
        Callable: the encode function to extract soft speech units and log probabilies using HubertSoft.
    """
    seg = segmenter(num_clusters=3, gamma=2, pretrained=pretrained, progress=progress)
    rhythm_model = rhythm_model_fine_grained(
        source_speaker=source_speaker,
        target_speaker=target_speaker,
        pretrained=pretrained,
        progress=progress,
    )
    time_stretcher = TimeStretcherFineGrained()
    vocoder = hifigan_generator(
        speaker=target_speaker,
        pretrained=pretrained,
        progress=progress,
        map_location=map_location,
    )
    return UrhythmicFine(seg, rhythm_model, time_stretcher, vocoder), encode


def urhythmic_global(
    source_speaker: str | None,
    target_speaker: str | None,
    pretrained: bool = True,
    progress: bool = True,
    map_location=None,
) -> Tuple[UrhythmicGlobal, Callable]:
    """Urhythmic (Global), a voice and rhythm conversion system that does not require text or parallel data.

    Available speakers:
        VCTK: p228, p268, p225, p232, p257, p231.
        LJSpeech.

    Args:
        source_speaker (None | str): the source speaker. None to fit your own source speaker or a selection from the available speakers.
        target_speaker (None | str): the target speaker. None to fit your own source speaker or a selection from the available speakers.
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.
        map_location: function or a dict specifying how to remap storage locations (see torch.load)

    Returns:
        UrhythmicFine: the Fine-Grained Urhythmic model.
        Callable: the encode function to extract soft speech units and log probabilies using HubertSoft.
    """
    seg = segmenter(num_clusters=3, gamma=2, pretrained=pretrained, progress=progress)
    rhythm_model = rhythm_model_global(
        source_speaker=source_speaker,
        target_speaker=target_speaker,
        pretrained=pretrained,
        progress=progress,
    )
    time_stretcher = TimeStretcherGlobal()
    vocoder = hifigan_generator(
        speaker=target_speaker,
        pretrained=pretrained,
        progress=progress,
        map_location=map_location,
    )
    return UrhythmicGlobal(seg, rhythm_model, time_stretcher, vocoder), encode
