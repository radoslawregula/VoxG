import copy
from math import ceil

import librosa
import numpy as np
import pandas as pd
import pysptk
import pyworld as pw

from src.data.soundutils import frequency_to_pitch
from src.utils.helpers import interpolate_inf


class Features:
    COL_T_START = 't_start'
    COL_T_END = 't_end'
    COL_PHONEME = 'phoneme'
    COL_REPEATS = 'repeats'

    def __init__(self):
        self.phonemes = []

    @staticmethod
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

    def signal_to_features(self, signal: np.ndarray, sampling_rate: float, 
                           frame_period_samples: int = 256) -> np.ndarray:
        frame_period = (frame_period_samples / sampling_rate) / 10**(-3)  # in miliseconds 
        f_zero, spectral_env, aperiod = pw.wav2world(signal,
                                                    sampling_rate,
                                                    fft_size=1024,
                                                    frame_period=frame_period)
        aperiod = librosa.amplitude_to_db(aperiod)
        spectral_env = librosa.power_to_db(spectral_env)

        spectral_env = self.dimensionality_reduce(spectral_env, n_dim=60, alpha=0.45)
        aperiod = self.dimensionality_reduce(aperiod, n_dim=4, alpha=0.45)
        
        with np.errstate(divide='ignore'):
            f_zero, interpolation_mask = interpolate_inf(frequency_to_pitch(f_zero)) 

        features = np.hstack((spectral_env, aperiod, f_zero, interpolation_mask))  

        return features

    @staticmethod
    def _read_phoneme_file(txt_file: str) -> pd.DataFrame:
        return pd.read_csv(txt_file, sep=' ', header=None, 
                        names=['t_start', 't_end', 'phoneme'])

    @staticmethod
    def _seconds_to_subframe_idx(sec: float, sr: float, subframe: int) -> int:
        return ceil((sec * sr) / subframe)
    
    # TODO: inspect why that func is used
    @staticmethod
    def _substitute(phoneme: str) -> str:
        return 'Sil' if phoneme in ('sil', 'br', 'pau') else phoneme

    def process_phonemes_file(self, txt_file: str, sampling_rate: float, 
                              subframe: int = 256) -> np.ndarray:
        df = self._read_phoneme_file(txt_file)
        col_subset = [self.COL_T_START, self.COL_T_END]
        df[col_subset] = df[col_subset].applymap(lambda x: (
            self._seconds_to_subframe_idx(x, 
                                          sr=sampling_rate, 
                                          subframe=subframe)
        ))
        df[self.COL_PHONEME] = df[self.COL_PHONEME].apply(self._substitute)
        phonemes_here = df[self.COL_PHONEME].unique().tolist()

        if not self.phonemes:
            self.phonemes.extend(phonemes_here)
        else:
            self.phonemes.extend([pho for pho in phonemes_here 
                                  if pho not in self.phonemes])
        
        df[self.COL_PHONEME] = df[self.COL_PHONEME].apply(self.phonemes.index)
        df[self.COL_REPEATS] = df[self.COL_T_END].sub(df[self.COL_T_START])

        phoneme_per_subframe = np.concatenate(
            [np.repeat(row[self.COL_PHONEME], 
                       row[self.COL_REPEATS]) for _, row in df.iterrows()]
        )

        return phoneme_per_subframe
        


    