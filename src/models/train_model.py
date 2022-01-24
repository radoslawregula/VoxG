# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click
import tensorflow as tf

from src.data.normalizer import Normalizer
from src.data.splits import Split
from src.models.data_feeder import DataFeeder
from src.utils.config import read_config_section


def _prepare_data_feeder(cfg: dict) -> DataFeeder:
    split = Split(cfg)
    normalizer = Normalizer(cfg)
    data_feeder = DataFeeder(cfg, split, normalizer)

    return data_feeder

@click.command()
@click.option('-c', '--config', type=str)
def main(config: str):
    logger = logging.getLogger(__name__)
    logger.info(f'Tensorflow version: {tf.__version__}')
    logger.info('Initializing training...')

    cfg = read_config_section(config)
    feeder = _prepare_data_feeder(cfg)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main() # pylint: disable=no-value-for-parameter
