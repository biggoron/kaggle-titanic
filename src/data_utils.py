import pandas as pd 
import json
import logging
from dotmap import DotMap

from utils import get_basic_logger

logger = get_basic_logger()

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

def preprocess_data(df, train=True):
    df = one_hot_encode_class(df)
    df = one_hot_encode_sex(df)
    df = one_hot_encode_port(df)
    df = drop_useless_columns(df)
    df = coerce_type_to_float(df)
    return df

def one_hot_encode_class(df):
    encodings = pd.get_dummies(df.Pclass)
    encodings = encodings.rename(columns={1: "first_class", 2: "second_class", 3: "third_class"})
    df = df.drop(columns=['Pclass'])
    df = pd.concat([df, encodings], axis=1)
    return df

def one_hot_encode_sex(df):
    encodings = pd.get_dummies(df.Sex)
    df = df.drop(columns=['Sex'])
    df = pd.concat([df, encodings], axis=1)
    return df

def one_hot_encode_port(df):
    encodings = pd.get_dummies(df.Embarked)
    encodings = encodings.rename(columns={"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
    df = df.drop(columns=['Embarked'])
    df = pd.concat([df, encodings], axis=1)
    return df

def drop_useless_columns(df):
    df = df.drop(columns=['Ticket', 'Cabin', 'Name'])
    return df

def coerce_type_to_float(df):
    cols = df.select_dtypes(exclude=['float']).columns
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    return df
