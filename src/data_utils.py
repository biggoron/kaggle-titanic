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
    df = filter_name_title(df)
    df = add_cabin(df)
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

def check_title(title, name):
    if title in name:
        return 1
    else:
        return 0

def impute_other_title(name):
    colonel = "Col." in name
    doctor = "Dr." in name
    sir = "Sir." in name
    captain = "Capt." in name
    reverend = "Rev." in name
    lady = "Lady." in name
    major = "Major." in name
    if colonel or doctor or sir or captain or reverend or lady or major:
        return 1
    else:
        return 0
        

def filter_name_title(df):
    df["mister"] = df.Name.apply(lambda name: check_title("Mr.", name))
    df["master"] = df.Name.apply(lambda name: check_title("Master.", name))
    df["missus"] = df.Name.apply(lambda name: check_title("Mrs.", name))
    df["miss"] = df.Name.apply(lambda name: check_title("Miss.", name))
    df["other_title"] = df.Name.apply(lambda name: impute_other_title(name))
    df.drop(columns="Name")
    return df

def add_cabin(df):
    df.Cabin = df.Cabin.apply(lambda cabin: 0 if pd.isna(cabin) else 1)
    return df

def drop_useless_columns(df):
    df = df.drop(columns=['Ticket'])
    return df

def coerce_type_to_float(df):
    cols = df.select_dtypes(exclude=['float']).columns
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    return df
