import copy
import logging
from math import ceil
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import pysptk
import pyworld as pw

from src.data.soundutils import frequency_to_pitch, pitch_to_frequency
from src.utils.constants import IndexableConstants as idc
from src.utils.helpers import interpolate_inf


logger = logging.getLogger(__name__)


class Features:
    COL_T_START = 't_start'
    COL_T_END = 't_end'
    COL_PHONEME = 'phoneme'
    COL_REPEATS = 'repeats'

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

        mfsc = (10 / np.log(10)) * mfsc  # but this is a linear transformation??

        return mfsc
    
    @staticmethod
    def dimensionality_decode(encoded_input: np.ndarray, n_dim: int, alpha: float, 
                              fftlen: int = 2048) -> np.ndarray:
        encoded_input = (np.log(10) / 10) * encoded_input

        # This part is taken directly from NPSS
        input_mirror = np.fft.irfft(encoded_input)
        input_back = input_mirror[:, :n_dim]
        input_back[:, 0] /= 2
        input_back[:, -1] /= 2

        spectral_output = np.apply_along_axis(pysptk.mgc2sp, axis=1, arr=input_back, 
                                              alpha=alpha, fftlen=fftlen)
        spectral_output = (20.0 / np.log(10)) * np.real(spectral_output)

        return spectral_output

    def signal_to_features(self, signal: np.ndarray, sampling_rate: float, 
                           frame_period_samples: int = 256) -> np.ndarray:
        # in miliseconds
        frame_period = (frame_period_samples / sampling_rate) / 10**(-3)   
        f_zero, spectral_env, aperiod = pw.wav2world(signal,
                                                     sampling_rate,
                                                     # fft_size=1024,
                                                     frame_period=frame_period)
        spectral_env = librosa.power_to_db(spectral_env, amin=1e-30, top_db=None)
        aperiod = librosa.amplitude_to_db(aperiod, amin=1e-30, top_db=None)

        spectral_env = self.dimensionality_reduce(spectral_env, n_dim=60, alpha=0.45)
        aperiod = self.dimensionality_reduce(aperiod, n_dim=4, alpha=0.45)
        
        with np.errstate(divide='ignore'):
            f_zero, interpolation_mask = interpolate_inf(frequency_to_pitch(f_zero)) 

        features = np.hstack((spectral_env, aperiod, f_zero, interpolation_mask))  

        return features
    
    def features_to_signal(self, features: np.ndarray, sampling_rate: float, 
                           frame_period_samples: int = 256) -> np.ndarray:
        spectral_env, aperiod, f_zero, interpolation_mask = np.hsplit(features, 
                                                                      indices_or_sections=[60, 64, 65])
        f_zero = pitch_to_frequency(f_zero)
        f_zero[interpolation_mask.astype(np.bool)] = 0.0
        f_zero = f_zero.flatten()
        
        spectral_env = self.dimensionality_decode(spectral_env, n_dim=60, alpha=0.45)
        aperiod = self.dimensionality_decode(aperiod, n_dim=4, alpha=0.45)

        spectral_env = librosa.db_to_power(spectral_env)
        aperiod = librosa.db_to_amplitude(aperiod)

        frame_period = (frame_period_samples / sampling_rate) / 10**(-3)  # in miliseconds
        signal = pw.synthesize(f_zero, spectral_env, aperiod, fs=sampling_rate, 
                               frame_period=frame_period)
        
        return signal
    
    @staticmethod
    def wav_to_mcep(signal: np.ndarray, sampling_rate: float, 
                    alpha: float = 0.45, fft_size: int = None, mcep_size: int = 34, 
                    frame_period_samples: int = 256) -> np.ndarray:
        # in miliseconds
        frame_period = (frame_period_samples / sampling_rate) / 10**(-3)
        _, spectral, _ = pw.wav2world(signal, fs=sampling_rate,
                                      frame_period=frame_period, fft_size=fft_size)

        # Extract MCEP features
        mcep = pysptk.sptk.mcep(spectral, order=mcep_size, alpha=alpha, maxiter=0,
                                etype=1, eps=1.0E-8, min_det=0.0, itype=3)
        
        return mcep

    @staticmethod
    def _read_phoneme_file(txt_file: str) -> pd.DataFrame:
        return pd.read_csv(txt_file, sep=' ', header=None, 
                           names=['t_start', 't_end', 'phoneme'])

    @staticmethod
    def _seconds_to_subframe_idx(sec: float, sr: float, subframe: int) -> int:
        return ceil((sec * sr) / subframe)

    @staticmethod
    def _match_phoneme_to_index(phoneme: str) -> int:
        if phoneme in idc.PHONEMES:
            return idc.PHONEMES.index(phoneme)
        else:
            raise RuntimeError(f'Phoneme {phoneme} is not mapped in the static '
                               f'mapping structure. Add mapping to proceed.')
    
    # SIL = BR = PAU = no sound!
    @staticmethod
    def _substitute(phoneme: str) -> str:
        return 'Sil' if phoneme in ('sil', 'br', 'pau') else phoneme
    
    def _verify_monotonic_increase(self, data: pd.DataFrame) -> pd.DataFrame:
        for idx, row in data.iterrows():
            if idx == 0:
                previous_row = row
            else:
                if previous_row[self.COL_T_END] > row[self.COL_T_START]:
                    row[self.COL_T_START] = previous_row[self.COL_T_END]
                    data.iloc[idx] = row
                previous_row = row
        
        return data

    def process_phonemes_file(self, txt_file: str, sampling_rate: float, 
                              subframe: int = 256) -> np.ndarray:
        df = self._read_phoneme_file(txt_file)
        col_subset = [self.COL_T_START, self.COL_T_END]
        df[col_subset] = df[col_subset].applymap(lambda x: (
            self._seconds_to_subframe_idx(x, 
                                          sr=sampling_rate, 
                                          subframe=subframe)
        ))
        df = self._verify_monotonic_increase(df)
        df[self.COL_PHONEME] = df[self.COL_PHONEME].apply(self._substitute)
        df[self.COL_PHONEME] = df[self.COL_PHONEME].apply(self._match_phoneme_to_index)
        df[self.COL_REPEATS] = df[self.COL_T_END].sub(df[self.COL_T_START])
        df[self.COL_REPEATS] = df[self.COL_REPEATS].apply(lambda x: x if x >= 0 else 0)

        phoneme_per_subframe = np.concatenate(
            [np.repeat(row[self.COL_PHONEME], 
                       row[self.COL_REPEATS]) for _, row in df.iterrows()]
        )

        return phoneme_per_subframe
    
    @staticmethod
    def crop(vector: np.ndarray, diff: int) -> np.ndarray:
        if diff == 0:
            return vector
        if diff % 2 == 0:
            to_crop = int(diff / 2)
            return vector[to_crop:-to_crop]
        else:
            diff += 1
            to_crop = int(diff / 2)
            if to_crop > 1:
                return vector[to_crop:-(to_crop - 1)]
            else:
                return vector[to_crop:]
    
    @staticmethod
    def match_subframes(stft: np.ndarray, features: np.ndarray, 
                        phonemes: np.ndarray) -> Tuple[np.ndarray]:
        def _dim(vec: np.ndarray) -> int:
            return vec.shape[0]
        
        diff_stft = _dim(stft) - _dim(phonemes)
        diff_feats = _dim(features) - _dim(phonemes)

        if not diff_feats == diff_stft:
            logger.warning(f'Unexpected shape mismatch between STFT and feature '
                           f'matrix. Will attempt to match anyway...')
        stft = Features.crop(stft, diff_stft)
        features = Features.crop(features, diff_feats)
        
        if len(set(map(_dim, [stft, features, phonemes]))) != 1:
            logger.error('Subframe matching failed: unexpected '
                         'data dimensions.')
            raise SystemExit
        
        return stft, features
