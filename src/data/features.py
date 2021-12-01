import librosa
import numpy as np
import pyworld as pw

from src.data.soundutils import frequency_to_pitch

def signal_to_features(signal: np.ndarray, sampling_rate: float, 
                       frame_period_samples: int = 256) -> np.ndarray:
    frame_period = (frame_period_samples / sampling_rate) / 10**(-3)  # in miliseconds 
    f0, spectral_env, aperiod = pw.wav2world(signal,
                                             sampling_rate,
                                             fft_size=1024,
                                             frame_period=frame_period)
    aperiod = librosa.amplitude_to_db(aperiod)
    spectral_env = librosa.power_to_db(spectral_env)
    f0 = frequency_to_pitch(f0)

    # TODO: get rif of -infs

    return None
