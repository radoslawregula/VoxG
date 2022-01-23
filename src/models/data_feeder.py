import os
import random

import numpy as np

from src.data.datasets import DataProcessor
from src.data.normalizer import Normalizer
from src.data.splits import Split
from src.utils.constants import IndexableConstants as idc


class DataFeeder:
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
    def _to_singer_index(datapoint: str) -> int:
        return idc.SINGERS.index(os.path.basename(datapoint).split('-')[0])

    def train_data_generator(self):
        # This many files is used in each epoch, samples_per_file frames from each
        num_files = int(self.batch_size / self.blocks_per_file)

        for _ in range(self.per_epoch):
            features_ = []
            f0_ = []
            singers_ = []
            phonemes_ = []

            files_this_batch = random.sample(self.split.train_dataset.listfiles(), 
                                             k=num_files)
            for f in files_this_batch:
                # TOCHECK: this process does not guarantee 
                # exhausting the set of possible files.
                singer_idx = self._to_singer_index(f)
                f_features, f_phonemes, _ = DataProcessor.from_hdf5(f)
                n_features = self.normalizer.normalize(f_features)

                candidate_starting_samples = range(n_features.shape[0] - self.block_len)
                blocks_this_file = random.sample(candidate_starting_samples, 
                                                 k=self.blocks_per_file)
                
                for block in blocks_this_file:
                    features_.append(n_features[block:block+self.block_len])
                    phonemes_.append(f_phonemes[block:block+self.block_len])
                    f0_.append(n_features[block:block+self.block_len, -2])
                    singers_.append(singer_idx)
                
            features_ = np.array(features_)
            phonemes_ = np.array(phonemes_)
            f0_ = np.expand_dims(np.array(f0_), -1)
            singers_ = np.array(singers_)

            self.normalizer.assert_within_bounds(features_)

            yield features_, f0_, phonemes_, singers_

    def valid_data_generator(self):
        pass

