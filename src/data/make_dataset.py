# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import click

from src.data.datasets import DataOrganizer, DataProcessor
from src.utils.config import read_config_section


@click.command()
@click.option('-c', '--config', type=str)
def main(config: str):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data...')

    cfg = read_config_section(config)
    data_organizer = DataOrganizer(cfg['input_data'])
    data_organizer.search_data()

    data_processor = DataProcessor(cfg)
    data_processor.preprocess(data_organizer.organizer)


if __name__ == '__main__':
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
