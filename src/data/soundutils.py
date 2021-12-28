import numpy as np


def frequency_to_pitch(frequency: float) -> float:
    return 69.0 + 12 * np.log2(frequency / 440.0)


def pitch_to_frequency(pitch: float) -> float:
    return 440.0 * np.power(2, (pitch - 69.0) / 12) 