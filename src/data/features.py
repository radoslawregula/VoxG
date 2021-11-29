import numpy as np
import pyworld as pw

def signal_to_features(signal: np.ndarray, sampling_rate: float, 
                       frame_period_samples: int = 256) -> np.ndarray:
    frame_period = (frame_period_samples / sampling_rate) / 10**3  # in miliseconds 
    f0, spectral_env, aperiod = pw.wav2world(signal,
                                             sampling_rate,
                                             fft_size=1024,
                                             frame_period=frame_period)