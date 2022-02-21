import numpy as np

from typing import Union


def frequency_to_pitch(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 69.0 + 12 * np.log2(frequency / 440.0)


def pitch_to_frequency(pitch: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 440.0 * np.power(2, (pitch - 69.0) / 12)
