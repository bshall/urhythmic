# Urhythmic: Rhythm Modeling for Voice Conversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.06040)
[![demo](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat)](https://ubisoft-laforge.github.io/speech/urhythmic/)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bshall/urhythmic/blob/main/urhythmic_demo.ipynb)

Official repository for [Rhythm Modeling for Voice Conversion](https://arxiv.org/abs/2307.06040).
Audio samples can be found [here](https://ubisoft-laforge.github.io/speech/urhythmic/). 
Colab demo can be found [here](https://colab.research.google.com/github/bshall/urhythmic/blob/main/urhythmic_demo.ipynb).

**Abstract**: Voice conversion aims to transform source speech into a different target voice. However, typical voice conversion systems do not account for rhythm, which is an important factor in the perception of speaker identity. To bridge this gap, we introduce Urhythmic - an unsupervised method for rhythm conversion that does not require parallel data or text transcriptions. Using self-supervised representations, we first divide source audio into segments  approximating sonorants, obstruents, and silences. Then we model rhythm by estimating speaking rate or the duration distribution of each segment type. Finally, we match the target speaking rate or rhythm by time-stretching the speech segments.Experiments show that Urhythmic outperforms existing unsupervised methods in terms of quality and prosody.

Note: Urhythmic builds on soft speech units from our paper [A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion](https://github.com/bshall/soft-vc/).

## Example Usage

### Programmatic Usage

```python
import torch, torchaudio

# Load the HubertSoft content encoder (see https://github.com/bshall/hubert/)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()

# Select the source and target speakers
# Pretrained models are available for:
# VCTK: p228, p268, p225, p232, p257, p231.
# and LJSpeech.
source, target = "p231", "p225"

# Load Urhythmic (either urhythmic_fine or urhythmic_global)
urhythmic, encode = torch.hub.load(
    "bshall/urhythmic:main", 
    "urhythmic_fine", 
    source_speaker=source, 
    target_speaker=target,
)
urhythmic.cuda()

# Load the source audio
wav, sr = torchaudio.load("path/to/wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Convert to the target speaker
with torch.inference_mode():
    # Extract speech units and log probabilities
    units, log_probs = encode(hubert, wav)
    # Convert to the target speaker
    wav_ = urhythmic(units, log_probs)
```

### Script-Based Usage

```
usage: convert.py [-h] [--extension EXTENSION]
                  {urhythmic_fine,urhythmic_global} source-speaker target-speaker in-dir out-dir

Convert audio samples using Urhythmic.

positional arguments:
  {urhythmic_fine,urhythmic_global}
                        available models (Urhythmic-Fine or Urhythmic-Global).
  source-speaker        the source speaker: p228, p268, p225, p232, p257, p231, LJSpeech
  target-speaker        the target speaker: p228, p268, p225, p232, p257, p231, LJSpeech
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

options:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .wav).
```

## Training

Here we outline the training steps for [VCTK](https://datashare.ed.ac.uk/handle/10283/3443). However, it should be straight forward to extend the recipe to other datasets.

1. [Prepare the Dataset](### Step 1: Prepare the Dataset)
2. [Extract Soft Speech Units and Log Probabilities](### Step 2: Extract Soft Speech Units and Log Probabilities)
3. [Train the Segmenter](### Step 3: Train the Segmenter)
4. [Segmentation and Clustering](### Step 4: Segmentation and Clustering)
5. [Train the Rhythm Model](### Step 5: Train the Rhythm Model)
6. [Train or Finetune the Vocoder](### Step 6: Train or Finetune the Vocoder)

### Step 1: Prepare the Dataset

Download and extract [VCTK](https://datashare.ed.ac.uk/handle/10283/3443). Split the data into `train` and `dev` sets for a given speaker (e.g. `p225`). The resulting directory should have the following structure:

```
p225
    ├── dev
    │   ├── wavs
    │   │   ├── p225_001.wav
    │   │   ├── ...
    │   │   ├── p225_024.wav
    ├── train
    │   ├── wavs
    │   │   ├── p225_025.wav
    │   │   ├── ...
    │   │   ├── p225_366.wav

```

Next, resample the audio to 16kHz using the `resample.py` script.
Note that the script will replace each file with a 16kHz version so remember to copy your data if you want to keep the originals.

```
usage: resample.py [-h] [--sample_rate SAMPLE_RATE] in-dir

Segment an audio dataset.

positional arguments:
  in-dir                path to dataset directory.

options:
  -h, --help            show this help message and exit
  --sample_rate SAMPLE_RATE
                        target sample rate (defaults to 16000).
```

For example:

```
python resample.py path/to/p225
```

### Step 2: Extract Soft Speech Units and Log Probabilities

Encode the `dev` and `train` sets using HuBERT-Soft and the `encode.py` script:

```
usage: encode.py [-h] [--extension EXTENSION] in-dir out-dir

Encode an audio dataset into soft speech units and the log probabilities of the associated discrete units.

positional arguments:
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.

options:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        extension of the audio files (defaults to .wav).
```

for example:

```
python encode.py path/to/p225/dev/wavs path/to/p225/dev
```

At this point the directory tree should look as follows:

```
p225
    ├── dev
    │   ├── wavs
    │   ├── soft
    │   ├── logprobs
    ├── train
    │   ├── wavs
    │   ├── soft
    │   ├── logprobs
```

### Step 3: Train the Segmenter

Cluster the discrete speech units and identify the cluster id corresponding to sonorants, obstruents, and silences using the `train_segmenter.py` script:

```
usage: train_segmenter.py [-h] dataset-dir checkpoint-path

Cluster the codebook of discrete speech units and identify the cluster id
corresponding to sonorants, obstruents, and silences.

positional arguments:
  dataset-dir      path to the directory of segmented speech.
  checkpoint-path  path to save checkpoint.

options:
  -h, --help       show this help message and exit
```

for example:

```
python train_segmenter.py path/to/p225/dev/ path/to/checkpoints/segmenter.pt
```

### Step 4: Segmentation and Clustering

Segment the `dev` and `train` sets using the `segment.py` script.
Note, this script uses the [segmenter checkpoint](https://github.com/bshall/urhythmic/releases/tag/v0.1).
You'll need to adapt the script to use your own checkpoint.

```
usage: segment.py [-h] in-dir out-dir

Segment an audio dataset.

positional arguments:
  in-dir      path to the log probability directory.
  out-dir     path to the output directory.

options:
  -h, --help  show this help message and exit
```

for example:

```
python segment.py path/to/p225/dev/logprobs path/to/p225/dev/segments/
```

At this point the directory tree should look as follows:

```
p225
    ├── dev
    │   ├── wavs
    │   ├── soft
    │   ├── logprobs
    │   ├── segments
    ├── train
    │   ├── wavs
    │   ├── soft
    │   ├── logprobs
    │   ├── segments
```

### Step 5: Train the Rhythm Model

Train the fine-grained or global rhythm model using the `train_rhythm_model.py` script:

```
usage: train_rhythm_model.py [-h] {fine,global} dataset-dir checkpoint-path

Train the FineGrained or Global rhythm model.

positional arguments:
  {fine,global}    type of rhythm model (fine-grained or global).
  dataset-dir      path to the directory of segmented speech.
  checkpoint-path  path to save checkpoint.

options:
  -h, --help       show this help message and exit
```

for example:

```
python train_rhythm_model.py fine path/to/p225/train/segments path/to/checkpoints/rhythm-fine-p225.pt
```

### Step 6: Train or Finetune the Vocoder

Train or finetune the HiFiGAN vocoder. We recommend finetuning from the [LJSpeech checkpoint](https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-LJSpeech-ceb1368d.pt).


```
usage: train_vocoder.py [-h] [--resume RESUME] [--finetune | --no-finetune] dataset-dir checkpoint-dir

Train or finetune the HiFiGAN vocoder.

positional arguments:
  dataset-dir           path to the preprocessed data directory
  checkpoint-dir        path to the checkpoint directory

options:
  -h, --help            show this help message and exit
  --resume RESUME       path to the checkpoint to resume from
  --finetune, --no-finetune
                        whether to finetune

```

For example, to train from scratch:

```
python train_vocoder.py /path/to/p225 /path/to/checkpoints
```

To finetune, download the [LJSpeech checkpoint](https://github.com/bshall/urhythmic/releases/download/v0.1/hifigan-LJSpeech-ceb1368d.pt) and run:

```
python train_vocoder.py /path/to/p225 /path/to/checkpoints --resume hifigan-LJSpeech-ceb1368d.pt --finetune
```

## Citation

If you found this work helpful please consider citing our paper.