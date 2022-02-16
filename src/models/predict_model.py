# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click

from src.data.normalizer import Normalizer
from src.models.data_feeder import DataFeeder
from src.models.inference import Inference
from src.utils.config import read_config_section


def _prepare_inference_data_feeder(cfg: dict) -> DataFeeder:
    normalizer = Normalizer(cfg)
    data_feeder = DataFeeder(cfg, split=None, normalizer=normalizer)

    return data_feeder


@click.command()
@click.option('-c', '--config', type=str)
@click.option('-f', '--file', type=str)
@click.option('-m', '--model', type=str)
def main(config: str, file: str, model: str):
    logger = logging.getLogger(__name__)
    logger.info('Running inference...')

    cfg = read_config_section(config)
    feeder = _prepare_inference_data_feeder(cfg)
    infer = Inference(cfg, feeder)
    infer.get(file, model)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main() # pylint: disable=no-value-for-parameter
