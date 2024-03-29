import logging
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf

from src.data.datasets import DataProcessor
from src.data.features import Features
from src.models.data_feeder import DataFeeder
from src.models.generator import Generator
from src.utils.constants import IndexableConstants as idc
from src.utils.helpers import to_singer_index, to_narrow_limits
from src.visualization.visualizer import visualize_features

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, config: dict, feeder: DataFeeder):
        self.feeder: DataFeeder = feeder
        self.model = Generator(config['training'])
        
        config_sampling = config['sampling']
        self.sampling_rate = config_sampling['sampling_rate']
        self.hop_len = config_sampling['hop_len']

        self.processed_path = config['processed_data']
        self.gt_path = config['gt_data']
        self.generated_path = config['generated_data']
        self.visualizations_path = config['visualizations']
    
    def _validate_file(self, file_path: str) -> str:
        if os.path.isfile(file_path):
            return file_path
        else:
            alt_file_path = os.path.join(self.processed_path, file_path)
            if os.path.isfile(alt_file_path):
                return alt_file_path
            else:
                raise FileNotFoundError(f'No HDF5 file at {file_path}.')
    
    def get(self, file_path: str, model_path: str, visualize: bool):
        file_path = self._validate_file(file_path)
        features, phonemes, fourier = DataProcessor.from_hdf5(file_path)
        feats_manager = Features()
        
        if model_path is not None:
            logger.info('Generating ground truth and model prediction...')
            singer_idx = to_singer_index(file_path)
            modeled_features = self.run_inference_for_file(features, phonemes,
                                                           model_path, singer_idx)
            if visualize:
                visualize_features(features, modeled_features, 
                                   file_path, self.visualizations_path)
            signal_mdl = feats_manager.features_to_signal(modeled_features,
                                                          sampling_rate=self.sampling_rate,
                                                          frame_period_samples=self.hop_len)
            self.save_to_wav(signal=signal_mdl,
                             filepath=self._generate_result_wav_path(file_path))
            # return # debug
        else:
            logger.info('Generating ground truth, skipping model prediction...')
        
        signal_gt = feats_manager.features_to_signal(features,  # 28018x66 -> size to achieve
                                                     sampling_rate=self.sampling_rate,
                                                     frame_period_samples=self.hop_len)
        # Save ground truth to file
        self.save_to_wav(signal=signal_gt, 
                         filepath=self._generate_ground_truth_wav_path(file_path))
    
    def run_inference_for_file(self, f_features: np.ndarray, f_phonemes: np.ndarray, 
                               model_path: str, singer: int) -> np.ndarray:
        n_features = self.feeder.normalizer.normalize(f_features)
        n_f0 = n_features[:, -2]
        f_f0 = f_features[:, -2:]
        
        batches_f0, num_chunks = self.feeder.generate_batches(n_f0)
        batches_phonemes, _ = self.feeder.generate_batches(f_phonemes, 
                                                           one_hotize=True, 
                                                           depth=idc.N_PHONEMES)
        batches_phonemes = tf.squeeze(batches_phonemes)
        
        out_features_raw = []
        self.load_model(model_path)

        for batch_f0, batch_phonemes in zip(batches_f0, batches_phonemes):
            batch_singers = self.feeder.vectorize(singer, depth=idc.N_SINGERS)
            features_this_call = self.model(batch_f0, batch_phonemes, 
                                            batch_singers, training=False)
            out_features_raw.append(features_this_call.numpy())

        out_features_raw = np.array(out_features_raw)
        out_features_merged = self.feeder.merge_overlapadd(out_features_raw, num_chunks)
        out_features_merged = to_narrow_limits(out_features_merged)
        out_features_merged = self.feeder.normalizer.denormalize(out_features_merged)
        crop_to = min(n_features.shape[0], out_features_merged.shape[0])
        
        out_features_merged = out_features_merged[:crop_to]  # That ...
        
        out_features = np.concatenate([
            out_features_merged,
            f_f0[:crop_to]  # ...or that crop is not necessary: however, 
                            # we should be prepared for both scenarios
        ], axis=-1)

        return out_features
    
    def load_model(self, model_path):
        inputs = self.feeder.feed_input_definition()
        self.model(*inputs)
        try:
            self.model.load_weights(filepath=model_path,
                                    by_name=False)
        except Exception as e:
            logger.error(f'Failed to read model at {os.path.basename(model_path)}.')
            raise e

    def _generate_ground_truth_wav_path(self, file_path: str) -> str:
        filename = os.path.basename(file_path).replace('processed', 'ground-truth') \
                                              .replace('hdf5', 'wav')
        return os.path.join(self.gt_path, filename)
    
    def _generate_result_wav_path(self, file_path: str) -> str:
        filename = os.path.basename(file_path).replace('processed', 'generated') \
                                              .replace('hdf5', 'wav')
        return os.path.join(self.generated_path, filename)
                            
    def save_to_wav(self, signal: np.ndarray, filepath: str):
        try:
            sf.write(file=filepath, data=signal, samplerate=int(self.sampling_rate))
            logging.info(f'Saved to file {os.path.basename(filepath)}.')
        except Exception as exc:
            logger.error(f'Error saving to file: {filepath}')
            raise exc
