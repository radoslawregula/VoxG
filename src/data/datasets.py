import os
from typing import Dict, List

import librosa
import numpy as np
import scipy

from src.data.features import signal_to_features


class RawDataPoint:
    def __init__(self, txt: str, wav: str, id: str, is_sing: bool):
        self.txt = txt
        self.wav = wav
        self.id = id
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
        
        for dp in datapoints:
            txt = os.path.join(full_subpath, dp + '.txt')
            wav = os.path.join(full_subpath, dp + '.wav')
            is_sing = subpath == 'sing'
            out_dpoints.append(RawDataPoint(txt, wav, id=dp, is_sing=is_sing))
        
        return out_dpoints


class DataProcessor:
    def __init__(self, config: Dict):
        self.sampling_rate = config['sampling_rate']

    def preprocess(self, organizer: List[RawDataPoint]):
        """
        Runs the preprocessing procedures to transform raw data into HDF5 files. 
        """
        for point in organizer:
            audio, _ = librosa.load(point.wav, sr=self.sampling_rate, 
                                    mono=True, dtype=np.float64)
            fourier = librosa.stft(audio, n_fft=1024, hop_length=256,
                                   window=scipy.signal.windows.hann(1024))
            fourier = np.abs(fourier)
            features = signal_to_features(audio, sampling_rate=self.sampling_rate)