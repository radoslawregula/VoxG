import logging
import os
from typing import Dict

import yaml


def read_config_section(configfile: str, section: str = None) -> Dict:
    with open(configfile, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            logging.error(f'Error loading config file at {configfile}.')
            raise e
    
    if section is not None:
        assert section in config.keys(), \
            f'Config file has no section "{section}".'
        config = config[section]
    
    return config