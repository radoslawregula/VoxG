import glob
import json
import logging
import os
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetTypes:
    TRAIN = "train"
    VALID = "valid"

    LEGAL_TYPES = (
        TRAIN,
        VALID
    )


class SplitDataset:
    def __init__(self, name: str, csv_path: str, 
                 file_base_path: str, lazy_load: bool = False):
        self.name = name
        self.csv_path = csv_path
        self.file_base_path = file_base_path
        self.data = None

        if not lazy_load:
            self.read_data()
    
    def read_data(self):
        try:
            self.data = pd.read_csv(self.csv_path)
        except IOError as ioe:
            logger.error(f'Error encountered reading .csv file at {self.csv_path}.')
            raise ioe
    
    def listfiles(self) -> List[str]:
        if self.data is None:
            self.read_data()
        return list(map(lambda x: os.path.join(self.file_base_path, x), 
                        self.data['file_name'].tolist()))
    

class Split:
    def __init__(self, cfg: dict):
        self.splits_top_path = cfg['splits']
        self.id = int(cfg['data']['split_id'])

        self.train_dataset: SplitDataset = None
        self.valid_datasets: List[SplitDataset] = None

        self.read_split()
    
    def read_split(self):
        split_path = self._search_for_split_id()
        split_json = os.path.join(split_path, f'split-{self.id}.json')

        if not os.path.isfile(split_json):
            raise ValueError(f'No split definition JSON file at {split_json}.')

        with open(split_json, 'r') as jf:
            split_data = json.load(jf)
        
        self._validate_split_data(split_data)

        for subset in split_data['subsets']:
            if subset['type'] == DatasetTypes.TRAIN:
                csv_path = os.path.join(split_path, subset['csv_name'])
                self.train_dataset = SplitDataset(name=subset['name'],
                                                  csv_path=csv_path,
                                                  file_base_path=split_data['base_path'])
            elif subset['type'] == DatasetTypes.VALID:
                csv_path = os.path.join(split_path, subset['csv_name'])
                self.valid_datasets.append(SplitDataset(name=subset['name'],
                                                        csv_path=csv_path,
                                                        file_base_path=split_data['base_path']))
            else:
                if subset['type'] in DatasetTypes.LEGAL_TYPES:
                    raise NotImplementedError
                logger.warning(f'Encountered unsupported dataset type ' \
                               f'`{subset["type"]}`. This one will be skipped...')
                continue
        
        logger.info(f'Read split {split_data["number"]} of total size {split_data["size"]}.')


    def _search_for_split_id(self) -> str:
        existing_splits = glob.glob(self.splits_top_path)
        matched = list(filter(
            lambda x: int(os.path.basename(x).split('_')[1]) == self.id,
            existing_splits
        ))

        if len(matched) != 1:
            logger.error(f'Failed to unambiguously establish the split of id '
                         f'{self.id}. Did you mean any of {matched}?')
            raise SystemExit
        
        return matched[0]
    
    @staticmethod
    def _validate_split_data(data: dict):
        valid_subsets = list(filter(lambda x: x['type'] in DatasetTypes.LEGAL_TYPES,
                                    data['subsets']))
        # More than 2 subsets, one train and one valid necessary
        assert valid_subsets >= 2, 'Split needs to consist of ' \
                                   'at least two datasets.'
        # Only one train dataset (merging can be added)
        assert sum([ds['type'] == DatasetTypes.TRAIN for ds in valid_subsets]) == 1, \
            'Merging training datasets is currently not supported. ' \
            'Specify one training set per split.'
        # Names are distinct
        assert len(set([ds['name'] for ds in valid_subsets])) == len(valid_subsets), \
            'Dataset names have to be distinct.'