from torch import nn
from pathlib import Path
import sys
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.cleaning import CleanAudioComponent

class PreprocessorWithDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise = CleanAudioComponent(
            sample_rate=16000,
            highpass_cutoff=80,
            target_rms=0.1,
        )
        self.mel = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            normalize="per_feature",
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=128,
            n_fft=512,
            log=True,
            frame_splicing=1,
            dither=1.0e-05,
            pad_to=0,
            pad_value=0.0,
        )

    def forward(self, x):
        x = self.denoise(x)
        x = self.mel(x)
        return x