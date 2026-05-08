import audiomentations as AM


def get_training_augmentation(sample_rate: int = 16000):
    """Build the audiomentations augmentation pipeline for training.

    Each transform is added defensively so that missing optional dependencies
    (librosa for time/pitch, pydub/ffmpeg for MP3, pyroomacoustics for room sim)
    degrade gracefully rather than crashing at startup.
    """

    transforms = [
        AM.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        AM.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
        AM.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=5, p=0.15),
    ]

    _try_add(transforms, lambda: AM.AddColorNoise(min_snr_db=5, max_snr_db=30, p=0.3))
    _try_add(transforms, lambda: AM.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3, leave_length_unchanged=False))
    _try_add(transforms, lambda: AM.PitchShift(min_semitones=-2, max_semitones=2, p=0.3))
    # Simulates the MP3 compression artifacts present in dev/test splits
    _try_add(transforms, lambda: AM.Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.2))
    _try_add(transforms, lambda: AM.RoomSimulator(p=0.3))

    return AM.Compose(transforms)


def _try_add(transforms: list, factory) -> None:
    try:
        transforms.append(factory())
    except Exception:
        pass
