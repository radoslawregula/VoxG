from importlib.abc import Loader
import logging
import os
import re
from typing import Dict

from dotenv import find_dotenv, load_dotenv
import yaml

def _setup_env_parser():
    path_matcher = re.compile(r'.*?\${{ (\w+) }}.*?')

    def path_constructor(loader, node):
        value = node.value
        match = path_matcher.match(value)
        env_var = match.group()[4:-3]
        return os.path.abspath(os.environ.get(env_var))

    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)


def read_config_section(configfile: str, section: str = None) -> Dict:
    load_dotenv(find_dotenv())
    _setup_env_parser()

    with open(configfile, 'r') as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            logging.error(f'Error loading config file at {configfile}.')
            raise e
    
    if section is not None:
        assert section in config.keys(), \
            f'Config file has no section "{section}".'
        config = config[section]
    
    return config