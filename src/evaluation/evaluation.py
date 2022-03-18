import logging
import math

import librosa
import numpy as np

from src.data.features import Features

logger = logging.getLogger(__name__)

# Implementation of mel-cepstral distortion for WAVs is partially sourced from this repo:
# https://github.com/SamuelBroughton/Mel-Cepstral-Distortion

class Evaluation:
    def __init__(self, config: dict):
        self.sampling_rate = config['sampling_rate']

    @staticmethod
    def mel_cepstral_dist(features: tuple) -> float:
        dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = features[1] - features[0]
        
        return dB_const * np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))

    def compute(self, file_synth: str, file_gt: str):
        features = []
        for f in (file_synth, file_gt):
            audio, _ = librosa.load(f, sr=self.sampling_rate, 
                                    mono=True, dtype=np.float64)
            mgc = Features.wav_to_mcep(audio, self.sampling_rate)
            features.append(mgc)
                
        mcd_result = self.mel_cepstral_dist(features)
        logger.info(f'Mel-cepstral distortion result: {round(mcd_result, 6)} dB.')
        

