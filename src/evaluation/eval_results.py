# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click

from src.evaluation.evaluation import Evaluation
from src.utils.config import read_config_section


@click.command()
@click.option('-c', '--config', type=str)
@click.option('-fs', '--file-synthesized', type=str)
@click.option('-fgt', '--file-ground-truth', type=str)
def main(config: str, file_synthesized: str, file_ground_truth: str):
    logger = logging.getLogger(__name__)
    logger.info('Running evaluation...')

    cfg = read_config_section(config)

    eval = Evaluation(cfg['sampling'])
    eval.compute(file_synthesized, file_ground_truth)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main() # pylint: disable=no-value-for-parameter
