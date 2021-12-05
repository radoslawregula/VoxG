import logging
import os
from typing import Dict, List

import librosa
import numpy as np
import scipy

from src.data.features import Features


logger = logging.getLogger(__name__)


class RawDataPoint:
    def __init__(self, txt: str, wav: str, id_numeric: str, is_sing: bool):
        self.txt = txt
        self.wav = wav
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
            singer_path = os.path.join(self.data_path, singer)
            read = self._process_subfolder(singer_path, 'read')
            sing = self._process_subfolder(singer_path, 'sing')

            self.organizer.extend(read + sing)

    @staticmethod
    def _process_subfolder(top_path, subpath) -> List[RawDataPoint]:
        out_dpoints = []
        full_subpath = os.path.join(top_path, subpath)
        files_to_process = os.listdir(full_subpath)
        datapoints = list(set(map(lambda x: x.rsplit('.', maxsplit=1)[0],
                                  files_to_process)))
        datapoints = sorted(datapoints)
        
        for dp in datapoints:
            txt = os.path.join(full_subpath, dp + '.txt')
            wav = os.path.join(full_subpath, dp + '.wav')
            is_sing = subpath == 'sing'
            out_dpoints.append(RawDataPoint(txt, wav, 
                                            id_numeric=dp,
                                            is_sing=is_sing))
        
        return out_dpoints


class DataProcessor:
    def __init__(self, config: Dict):
        config_sampling = config['sampling']

        self.sampling_rate = config_sampling['sampling_rate']
        self.frame_len = config_sampling['frame_len']
        self.hop_len = config_sampling['hop_len']

        self.feats = Features()

    def preprocess(self, organizer: List[RawDataPoint]):
        """
        Runs the preprocessing procedures to transform raw data into HDF5 files. 
        """
        for num, point in enumerate(organizer):
            logger.info(f'Processing file {num + 1}/{len(organizer)}...')
            audio, _ = librosa.load(point.wav, sr=self.sampling_rate, 
                                    mono=True, dtype=np.float64)
            fourier = librosa.stft(audio, n_fft=self.frame_len, 
                                   hop_length=self.hop_len,
                                   window=scipy.signal.windows.hann(self.frame_len))
            fourier = np.abs(fourier)

            features = self.feats.signal_to_features(audio, 
                                                     sampling_rate=self.sampling_rate)
            phonemes = self.feats.process_phonemes_file(point.txt, 
                                                        sampling_rate=self.sampling_rate, 
                                                        subframe=self.hop_len)
            