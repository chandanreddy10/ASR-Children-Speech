import torch
import torchaudio


def clean_audio(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    highpass_cutoff: int = 80,
    target_rms: float = 0.1,
) -> torch.Tensor:
    """
    Applies:
    1. High-pass biquad filter
    2. RMS normalization

    Args:
        audio: Tensor [T] or [B, T]
    """

    # Ensure shape is [B, T]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # High-pass filter
    audio = torchaudio.functional.highpass_biquad(
        audio,
        sample_rate,
        highpass_cutoff,
    )

    # RMS normalization (per utterance)
    rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
    audio = audio * (target_rms / (rms + 1e-8))

    # Remove batch dim if originally 1D
    if audio.shape[0] == 1:
        audio = audio.squeeze(0)

    return audio