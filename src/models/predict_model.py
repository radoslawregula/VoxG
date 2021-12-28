# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click

from src.models.inference import Inference
from src.utils.config import read_config_section


@click.command()
@click.option('-c', '--config', type=str)
@click.option('-f', '--file', type=str)
@click.option('-skp', '--skip-prediction', is_flag=True)
def main(config: str, file: str, skip_prediction: bool):
    logger = logging.getLogger(__name__)
    logger.info('Running inference...')
    if skip_prediction:
        logger.info('Generating ground truth, skipping model prediction.')

    cfg = read_config_section(config)
    infer = Inference(cfg)
    infer.get(file, skip_prediction)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
