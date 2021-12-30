import glob
import os

import numpy as np

from src.data.datasets import DataProcessor


class Normalizer:
    def __init__(self, config: dict):
        self.processed_path = config['processed_data']

    @staticmethod
    def simple_normalize(signal: np.ndarray, s_min: np.ndarray, 
                         s_max: np.ndarray) -> np.ndarray:
        return (signal - s_min) / (s_max - s_min)

    @staticmethod
    def imput_f_zero(f_zero: np.ndarray) -> np.ndarray:
        median = np.median(f_zero[f_zero > 0.0])
        f_zero[f_zero == 0.0] = median

        return f_zero
    
    def normalize_f_zero(self, f_zero: np.ndarray) -> np.ndarray:
        # TODO: s_min, s_max
        return self.simple_normalize(self.imput_f_zero(f_zero), s_min=None, s_max=None)
    
    def get_normalization_data(self):
        iterfiles = glob.glob(os.path.join(self.processed_path, '*-processed.hdf5'))
        # Why is that used?
        iterfiles = list(filter(lambda x: 'KENN' not in x, iterfiles))
        max_values = []
        min_values = []
        
        for file in iterfiles:
            features, _, _ = DataProcessor.from_hdf5(file)
            # F0 is a penultimate feature in the feature matrix
            features[:, -2] = self.imput_f_zero(features[:, -2])
            max_values.append(np.amax(features, axis=0))
            min_values.append(np.amin(features, axis=0))
        
        # TODO: combine max min values, get global min max, save to hdf5

    def norm_data_to_hdf5(self):
        pass