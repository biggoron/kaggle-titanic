from .data_utils import load_config, load_data, hello_world

config_path = "config/main_config.json"
config = load_config(config_path)
train_df, test_df = load_data(config.data)
hello_world(train_df)
hello_world(test_df)
