import glob
import os

import h5py
import logging
import numpy as np

from src.data.datasets import DataProcessor

logger = logging.getLogger(__name__)


class Normalizer:
    DSET_MAX = 'features_max'
    DSET_MIN = 'features_min'

    def __init__(self, config: dict):
        self.processed_path = config['processed_data']
        normalization_path = config['normalization_data']
        self.hdf5_fpath = os.path.join(normalization_path, 'nus_normalization.hdf5')

        self.features_max: np.ndarray = None
        self.features_min: np.ndarray = None

        # Read or calculate normalization data
        if os.path.isfile(self.hdf5_fpath):
            self.normalizer_data_from_hdf5()
        else:
            self.calculate_normalization_data(max_lower_bound=0.0,
                                              min_higher_bound=1000.0)

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
    
    def calculate_normalization_data(self, max_lower_bound: float = None,
                                     min_higher_bound: float = None):
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

        # Prepare vector of max feature elements
        max_values_combined = np.vstack(max_values)
        max_values_global = np.amax(max_values_combined, axis=0)
        if max_lower_bound is not None:
            max_values_global = np.where(max_values_global <= max_lower_bound,
                                         max_lower_bound,
                                         max_values_global)

        # Prepare vector of min feature elements
        min_values_combined = np.vstack(min_values)
        min_values_global = np.amin(min_values_combined, axis=0)
        if min_higher_bound is not None:
            min_values_global = np.where(min_values_global >= min_higher_bound,
                                         min_higher_bound,
                                         min_values_global)

        self.features_max = max_values_global
        self.features_min = min_values_global

        self.normalizer_data_to_hdf5()

    def normalizer_data_to_hdf5(self):
        try:
            with h5py.File(self.hdf5_fpath, 'w') as h5file:
                h5file.create_dataset(name=Normalizer.DSET_MAX,
                                      shape=self.features_max.shape,
                                      data=self.features_max, dtype=np.float64)
                h5file.create_dataset(name=Normalizer.DSET_MIN,
                                      shape=self.features_min.shape,
                                      data=self.features_min, dtype=np.float64)
            logger.info(f'Normalization HDF5 saved in {os.path.basename(self.hdf5_fpath)}.')
        except Exception:
            logger.error(f'Error saving {os.path.basename(self.hdf5_fpath)}.')
            return

    def normalizer_data_from_hdf5(self):
        try:
            with h5py.File(self.hdf5_fpath, 'r') as h5file:
                self.features_max = h5file.get(Normalizer.DSET_MAX)[()]
                self.features_min = h5file.get(Normalizer.DSET_MIN)[()]
            logger.info(f'Normalization HDF5 read from {os.path.basename(self.hdf5_fpath)}.')
        except Exception:
            logger.error(f'Error reading from {os.path.basename(self.hdf5_fpath)}')
            raise Exception
