import pandas as pd 
import json
import logging
from dotmap import DotMap

logger = logging.getLogger("basic")

def load_data(data_config):
    raw_train_dataset = pd.read_csv(data_config.train_path)
    raw_test_dataset = pd.read_csv(data_config.test_path)
    return raw_train_dataset, raw_test_dataset

def load_conf(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = DotMap(config)
    return config

def hello_world(data):
    logger.debug(f"Dataframe shape: {data.shape}")
    return None
