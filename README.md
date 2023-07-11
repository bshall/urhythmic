# Urhythmic: Rhythm Modeling for Voice Conversion

Official repository for [Rhythm Modeling for Voice Conversion](). Audio samples can be found [here](https://ubisoft-laforge.github.io/speech/urhythmic/). Colab demo can be found [here]().

**Abstract**: Voice conversion aims to transform source speech into a different target voice.  However, typical voice conversion systems do not account for rhythm, which is an important factor in the perception of speaker identity. To bridge this gap, we introduce Urhythmic - an unsupervised method for rhythm conversion that does not require parallel data or text transcriptions. Using self-supervised representations, we first divide source audio into segments  approximating sonorants, obstruents, and silences. Then we model rhythm by estimating speaking rate or the duration distribution of each segment type. Finally, we match the target speaking rate or rhythm by time-stretching the speech segments.Experiments show that Urhythmic outperforms existing unsupervised methods in terms of quality and prosody.

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
wav = source.unsqueeze(0).cuda()

# Convert to the target speaker
with torch.inference_mode():
    # Extract speech units and log probabilities
    units, log_probs = encode(hubert, wav)
    # Convert to the target speaker
    wav_ = urhythmic(units, log_probs)
```