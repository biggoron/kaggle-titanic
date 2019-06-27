import pandas as pd
import xgboost as xgb
from data_utils import load_conf, load_data, hello_world, preprocess_data
from train import train_booster, eval_booster
from utils import get_basic_logger

logger = get_basic_logger()

def round_pred(pred):
    if pred >= 0.5:
        return 1
    else:
        return 0

if __name__ == '__main__':
    config_path = "config/main_config.json"
    config = load_conf(config_path)
    train_df, test_df = load_data(config.data)
    train_df = preprocess_data(train_df, train=True)
    test_df = preprocess_data(test_df, train=False)
    auc = eval_booster(train_df)
    logger.info(f"AUC over 5 folds: {auc}")
    booster = train_booster(train_df)
    test_dataset = xgb.DMatrix(test_df.values)
    predictions = booster.predict(test_dataset)
    predictions = [round_pred(pred) for pred in predictions]

    submission = pd.DataFrame({
        "PassengerId": test_df.PassengerId,
        "Survived": predictions})

    submission = submission.astype(int)
    submission.to_csv("submission.csv", index=False)
