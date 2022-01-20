import logging
import os

import numpy as np
import soundfile as sf

from src.data.datasets import DataProcessor
from src.data.features import Features

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, config: dict):
        config_sampling = config['sampling']
        
        self.sampling_rate = config_sampling['sampling_rate']
        self.hop_len = config_sampling['hop_len']

        self.processed_path = config['processed_data']
        self.gt_path = config['gt_data']
        self.generated_data = config['generated_data']
    
    def _validate_file(self, file_path: str) -> str:
        if os.path.isfile(file_path):
            return file_path
        else:
            alt_file_path = os.path.join(self.processed_path, file_path)
            if os.path.isfile(alt_file_path):
                return alt_file_path
            else:
                raise FileNotFoundError(f'No HDF5 file at {file_path}.')
    
    def get(self, file_path: str, skip_prediction: bool):
        file_path = self._validate_file(file_path)
        features, phonemes, fourier = DataProcessor.from_hdf5(file_path)
        feats_manager = Features()
        
        signal_gt = feats_manager.features_to_signal(features, 
                                                     sampling_rate=self.sampling_rate,
                                                     frame_period_samples=self.hop_len)
        
        if not skip_prediction:
            # predict from model
            pass
        
        # Save ground truth to file
        self.save_to_wav(signal=signal_gt, 
                         filepath=self._generate_ground_truth_wav_path(file_path))

    def _generate_ground_truth_wav_path(self, file_path: str) -> str:
        filename = os.path.basename(file_path).replace('processed', 'ground-truth') \
                                              .replace('hdf5', 'wav')
        return os.path.join(self.gt_path, filename)
                            
    def save_to_wav(self, signal: np.ndarray, filepath: str):
        try:
            sf.write(file=filepath, data=signal, samplerate=int(self.sampling_rate))
            logging.info(f'Saved to file {os.path.basename(filepath)}.')
        except Exception as exc:
            logger.error(f'Error saving to file: {filepath}')
            raise exc
