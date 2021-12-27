import os

from src.data.datasets import DataProcessor
from src.data.features import Features

class Inference:
    def __init__(self, config: dict):
        self.config = config
    
    def _validate_file(self, file_path: str) -> str:
        if os.path.isfile(file_path):
            return file_path
        else:
            alt_file_path = os.path.join(self.config['processed_data'], file_path)
            if os.path.isfile(alt_file_path):
                return alt_file_path
            else:
                raise FileNotFoundError(f'No HDF5 file at {file_path}.')
    
    def get(self, file_path: str, ground_truth: bool):
        file_path = self._validate_file(file_path)
        features, phonemes, fourier = DataProcessor.from_hdf5(file_path)
        feats_manager = Features()
        
        if ground_truth:
            # generate WAV from features 
            signal = feats_manager.features_to_signal(features)
        else:
            # predict and generate WAV
            pass