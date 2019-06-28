import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

def eval_booster(dataframe):
    kf = KFold(n_splits=5)
    labels = dataframe.Survived.values
    features = dataframe.drop(columns=["Survived"]).values
    accuracy_buffer = 0
    for train_index, val_index in kf.split(features):
        features_train, features_val = features[train_index], features[val_index]
        labels_train, labels_val = labels[train_index], labels[val_index]
        features = dataframe.drop(columns=["Survived"]).values
        booster = XGBClassifier(max_depth=4, min_child_weight=3, subsample=1)
        booster.fit(
            features_train,
            labels_train,
            eval_metric="error",
            eval_set=[(features_val, labels_val)],
            verbose=True)
        val_predictions =  booster.predict(features_val)
        val_predictions = [round(value) for value in val_predictions]
        accuracy = accuracy_score(labels_val, val_predictions)
        accuracy_buffer += accuracy
    return accuracy_buffer / 5.0

def train_booster(dataframe):
    print(dataframe.columns)
    labels = dataframe.Survived.values
    features = dataframe.drop(columns=["Survived"]).values
    booster = XGBClassifier(max_depth=4, min_child_weight=3, subsample=1)
    booster.fit(
        features,
        labels,
        )
    return booster
