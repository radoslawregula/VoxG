import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src.data.datasets import DataProcessor
from src.data.normalizer import Normalizer
from src.data.splits import Split
from src.utils.constants import IndexableConstants as idc
from src.utils.helpers import to_singer_index


class DataFeeder:
    PAD_CONST = 1e-12

    def __init__(self, cfg: dict, split: Split, normalizer: Normalizer):
        self.split: Split = split
        self.normalizer: Normalizer = normalizer

        cfgt = cfg['training']
        self.batch_size = cfgt['batch_size']
        self.per_epoch = cfgt['per_epoch']
        self.per_epoch_valid = cfgt['per_epoch_valid']
        self.blocks_per_file = cfgt['blocks_per_file']
        self.block_len = cfgt['block_len']

    @staticmethod
    def _one_hot(vector: np.ndarray, n_classes: int) -> np.ndarray:
        return np.identity(n_classes)[vector]

    def train_data_generator(self):
        t_files = self.split.train_dataset.listfiles()
        t_rng = np.random.default_rng()  # always random

        train_generator = self._generator(files=t_files, random_generator=t_rng, 
                                          iterations=self.per_epoch)
        for features_, f0_, phonemes_, singers_ in train_generator:
            yield features_, f0_, phonemes_, singers_  # Gen-in-gen
        
    def valid_data_generator(self):
        v_files = []
        for valid_set in self.split.valid_datasets:
            v_files.extend(valid_set.listfiles())
        
        v_rng = np.random.default_rng(seed=49)  # reproducible

        valid_generator = self._generator(files=v_files, random_generator=v_rng,
                                          iterations=self.per_epoch_valid)
        for features_, f0_, phonemes_, singers_ in valid_generator:
            yield features_, f0_, phonemes_, singers_  # Gen-in-gen

    def _generator(self, files: List, random_generator: np.random.Generator, 
                   iterations: int):
        # This many files is used in each epoch, samples_per_file frames from each
        num_files = int(self.batch_size / self.blocks_per_file)

        for _ in range(iterations):
            features_ = []
            f0_ = []
            singers_ = []
            phonemes_ = []

            files = np.array(files)
            try:
                files_this_batch = random_generator.choice(files,
                                                           size=num_files, 
                                                           replace=False).tolist()
            except ValueError:
                files_this_batch = np.concatenate([
                    files, 
                    random_generator.choice(files, size=(num_files-len(files)), 
                                            replace=False).tolist()
                ])

            for f in files_this_batch:
                # TOCHECK: this process does not guarantee 
                # exhausting the set of possible files.
                singer_idx = to_singer_index(f)
                f_features, f_phonemes, _ = DataProcessor.from_hdf5(f)
                n_features = self.normalizer.normalize(f_features)

                cand_start = np.arange(n_features.shape[0] - self.block_len)
                blocks_this_file = random_generator.choice(cand_start, 
                                                           size=self.blocks_per_file, 
                                                           replace=False).tolist()
                
                for block in blocks_this_file:
                    _feats_block = n_features[block:block+self.block_len]
                    # Drop two last columns as they hold F0 data
                    features_.append(_feats_block[:, :-2])
                    # Extract F0 data
                    f0_.append(_feats_block[:, -2])
                    phonemes_.append(f_phonemes[block:block+self.block_len])
                    singers_.append(singer_idx)
                
            features_ = np.array(features_)
            phonemes_ = np.array(phonemes_)
            phonemes_ = self._one_hot(phonemes_, n_classes=idc.N_PHONEMES)
            singers_ = np.array(singers_)
            singers_ = self._one_hot(singers_, n_classes=idc.N_SINGERS)
            f0_ = np.expand_dims(np.array(f0_), -1)

            self.normalizer.assert_within_bounds(features_)

            # To tensors
            features_ = tf.convert_to_tensor(features_, dtype=tf.float32)
            f0_ = tf.convert_to_tensor(f0_, dtype=tf.float32)
            phonemes_ = tf.convert_to_tensor(phonemes_, dtype=tf.float32)
            singers_ = tf.convert_to_tensor(singers_, dtype=tf.float32)
            
            yield features_, f0_, phonemes_, singers_
    
    def vectorize(self, index: int, depth: int) -> tf.Tensor:
        vector = np.repeat(index, self.batch_size)
        vector = self._one_hot(vector, n_classes=depth)

        return tf.convert_to_tensor(vector)
    
    def generate_batches(self, signal: np.ndarray, one_hotize: bool = False, 
                         depth: int = None) -> Tuple[tf.Tensor, int]:
        # Collect variables to use in the context
        block_overlap = int(self.block_len / 2)
        signal_length = signal.shape[0]

        num_blocks = int(signal_length / block_overlap)
        pad_blocks = block_overlap - (signal_length % block_overlap)
        signal = np.pad(signal, (0, pad_blocks), mode='constant', 
                        constant_values=self.PAD_CONST)
        batches_flattened = np.stack([
            signal[idx * block_overlap:(idx * block_overlap) + self.block_len] 
            for idx in range(num_blocks)
            ])     
        
        pad_batches = self.batch_size - (batches_flattened.shape[0] % self.batch_size)
        batches_flattened = np.concatenate([
            batches_flattened, np.ones((pad_batches, self.block_len)) * self.PAD_CONST
        ])
        
        batches = batches_flattened.reshape((-1, self.batch_size, self.block_len, 1))
        if one_hotize:
            batches = self._one_hot(batches.astype(np.int16), n_classes=depth)
        batches = tf.convert_to_tensor(batches)

        return batches, num_blocks
    
    def feed_input_definition(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        input_f0 = tf.keras.Input(shape=(self.block_len, 1))
        input_phonemes = tf.keras.Input(shape=(self.block_len, idc.N_PHONEMES))
        input_singers = tf.keras.Input(shape=(idc.N_SINGERS,))

        return input_f0, input_phonemes, input_singers
