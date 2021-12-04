import copy
import librosa
import numpy as np
import pysptk
import pyworld as pw

from src.data.soundutils import frequency_to_pitch
from src.utils.helpers import interpolate_inf


def dimensionality_reduce(spectral_input: np.ndarray, n_dim: int, alpha: float, 
                          noise_floor: float = -120.0) -> np.ndarray:
    mcep = np.apply_along_axis(pysptk.mcep, axis=1, arr=spectral_input, 
                               order=n_dim-1, alpha=alpha, maxiter=0, etype=1,
                               eps=10**(noise_floor/10), min_det=0.0, itype=1)
    # This part is taken directly from NPSS
    scale_mcep = copy.copy(mcep)
    scale_mcep[:, 0] *= 2
    scale_mcep[:, -1] *= 2
    mirror = np.hstack([scale_mcep[:, :-1], scale_mcep[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc
    

def signal_to_features(signal: np.ndarray, sampling_rate: float, 
                       frame_period_samples: int = 256) -> np.ndarray:
    frame_period = (frame_period_samples / sampling_rate) / 10**(-3)  # in miliseconds 
    f0, spectral_env, aperiod = pw.wav2world(signal,
                                             sampling_rate,
                                             fft_size=1024,
                                             frame_period=frame_period)
    aperiod = librosa.amplitude_to_db(aperiod)
    spectral_env = librosa.power_to_db(spectral_env)

    spectral_env = dimensionality_reduce(spectral_env, n_dim=60, alpha=0.45)
    aperiod = dimensionality_reduce(aperiod, n_dim=4, alpha=0.45)
    
    with np.errstate(divide='ignore'):
        f0, interpolation_mask = interpolate_inf(frequency_to_pitch(f0)) 

    features = np.hstack((spectral_env, aperiod, f0, interpolation_mask))  

    return features
