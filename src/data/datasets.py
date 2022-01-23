import logging
import os
from typing import Dict, List, Tuple

import h5py
import librosa
import numpy as np
import scipy

from src.data.features import Features

logger = logging.getLogger(__name__)


class RawDataPoint:
    def __init__(self, txt: str, wav: str, singer_name: str, 
                 id_numeric: str, is_sing: bool):
        self.txt = txt
        self.wav = wav
        self.singer_name = singer_name
        self.id_numeric = id_numeric
        self.is_sing = is_sing


class DataOrganizer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.organizer = []
    
    def search_data(self):
        data_path_contents = os.listdir(self.data_path)
        data_path_contents = list(map(lambda x: os.path.join(self.data_path, x),
                                      data_path_contents))
        singers = list(filter(os.path.isdir, data_path_contents))

        for singer in singers:
            read = self._process_subfolder(singer, 'read')
            sing = self._process_subfolder(singer, 'sing')

            self.organizer.extend(read + sing)

    @staticmethod
    def _process_subfolder(singer: str, subpath: str) -> List[RawDataPoint]:
        out_dpoints = []
        full_subpath = os.path.join(singer, subpath)
        files_to_process = os.listdir(full_subpath)
        datapoints = list(set(map(lambda x: x.rsplit('.', maxsplit=1)[0],
                                  files_to_process)))
        datapoints = sorted(datapoints)
        
        for dp in datapoints:
            txt = os.path.join(full_subpath, dp + '.txt')
            wav = os.path.join(full_subpath, dp + '.wav')
            is_sing = subpath == 'sing'
            out_dpoints.append(RawDataPoint(txt, wav, 
                                            singer_name=os.path.basename(singer),
                                            id_numeric=dp,
                                            is_sing=is_sing))
        
        return out_dpoints


class DataProcessor:
    DSET_FEATS = 'features'
    DSET_PHO = 'phonemes'
    DSET_STFT = 'fourier'

    def __init__(self, config: Dict):
        config_sampling = config['sampling']

        self.sampling_rate = config_sampling['sampling_rate']
        self.frame_len = config_sampling['frame_len']
        self.hop_len = config_sampling['hop_len']

        self.feats = Features()

        self.output_path = config['processed_data']

    def preprocess(self, organizer: List[RawDataPoint]):
        """
        Runs the preprocessing procedures to transform raw data into HDF5 files. 
        """
        for num, point in enumerate(organizer):
            logger.info(f'Processing file {num + 1}/{len(organizer)}...')
            
            hdf5_fpath = os.path.join(self.output_path, 
                                      f'{point.singer_name}-{point.id_numeric}-' \
                                      f'{"sing" if point.is_sing else "read"}-' \
                                      f'processed.hdf5') 
            if os.path.isfile(hdf5_fpath):
                logger.info('Already processed.')
                continue

            audio, _ = librosa.load(point.wav, sr=self.sampling_rate, 
                                    mono=True, dtype=np.float64)
            fourier = librosa.stft(audio, n_fft=self.frame_len, 
                                   hop_length=self.hop_len,
                                   window=scipy.signal.windows.hann(self.frame_len))
            fourier = np.abs(fourier).transpose()

            features = self.feats.signal_to_features(audio, 
                                                     sampling_rate=self.sampling_rate)
            phonemes = self.feats.process_phonemes_file(point.txt, 
                                                        sampling_rate=self.sampling_rate, 
                                                        subframe=self.hop_len)
            fourier,features = self.feats.match_subframes(fourier, 
                                                          features, 
                                                          phonemes)
            self.to_hdf5(hdf5_fpath, features, phonemes, fourier)
        
        logger.info(f'Preprocessed data saved to {self.output_path}.')

    @staticmethod    
    def to_hdf5(hdf5_fpath: str, features_to_save: np.ndarray, 
                phonemes_to_save: np.ndarray, fourier_to_save: np.ndarray):
        try:
            with h5py.File(hdf5_fpath, 'w') as h5file:
                h5file.create_dataset(name=DataProcessor.DSET_FEATS, 
                                      shape=features_to_save.shape,
                                      data=features_to_save, dtype=np.float64)
                h5file.create_dataset(name=DataProcessor.DSET_PHO, 
                                      shape=phonemes_to_save.shape,
                                      data=phonemes_to_save, dtype=np.int16)
                h5file.create_dataset(name=DataProcessor.DSET_STFT, 
                                      shape=fourier_to_save.shape,
                                      data=fourier_to_save, dtype=np.float64)
            
            logger.info(f'HDF5 saved in {os.path.basename(hdf5_fpath)}.')
        except Exception:
            logger.error(f'Error saving {os.path.basename(hdf5_fpath)}. Skipping...')
            return
    
    @staticmethod
    def from_hdf5(hdf5_fpath: str) -> Tuple[np.ndarray]:
        if not os.path.isfile(hdf5_fpath):
            raise FileNotFoundError(f"No HDF5 file to use at {hdf5_fpath}.")
        try:
            with h5py.File(hdf5_fpath, 'r') as h5file:
                features_read = h5file.get(DataProcessor.DSET_FEATS)[()]
                phonemes_read = h5file.get(DataProcessor.DSET_PHO)[()]
                fourier_read = h5file.get(DataProcessor.DSET_STFT)[()]
            logger.info(f'HDF5 read from {os.path.basename(hdf5_fpath)}.')
        except Exception as e:
            logger.error(f'Error reading from {os.path.basename(hdf5_fpath)}')
            raise e
        
        return features_read, phonemes_read, fourier_read
