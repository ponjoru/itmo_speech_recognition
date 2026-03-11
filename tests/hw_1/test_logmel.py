import torch
import torchaudio
from matplotlib import pyplot as plt

from hw_1.melbanks import LogMelFilterBanks


def test_logmel():
    signal, sr = torchaudio.load("assets/hw_1/1aeef15e_nohash_0.wav")
    
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000
    
    hop_length = 160
    n_mels = 80
    
    m1 = torchaudio.transforms.MelSpectrogram(hop_length=hop_length, n_mels=n_mels)
    m2 = LogMelFilterBanks(hop_length=hop_length, n_mels=n_mels)
    
    logmelspec1 = torch.log(m1(signal) + 1e-6)
    logmelspec2 = m2(signal)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    axes[0].imshow(logmelspec1[0].detach().numpy(), aspect="auto", origin="lower")
    axes[0].set_title("Log-Mel Spectrogram (torchaudio)")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Mel bins")
    axes[1].imshow(logmelspec2[0].detach().numpy(), aspect="auto", origin="lower")
    axes[1].set_title("Log-Mel Spectrogram (custom)")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mel bins")
    fig.savefig("assets/hw_1/logmelspec.png")

    assert torch.isfinite(logmelspec1).all(), "Log mel spectrogram from torchaudio contains non-finite values"
    assert torch.isfinite(logmelspec2).all(), "Log mel spectrogram from custom implementation contains non-finite values"
    assert logmelspec1.shape == logmelspec2.shape, f"Shapes of log mel spectrograms do not match: {logmelspec1.shape} vs {logmelspec2.shape}"
    assert torch.allclose(logmelspec1, logmelspec2), "Log mel spectrograms from torchaudio and custom implementation do not match"
    