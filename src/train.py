import xgboost as xgb
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

def eval_booster(dataframe):
    kf = KFold(n_splits=5)
    labels = dataframe.Survived.values
    features = dataframe.drop(columns=["Survived"]).values
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 3
    auc_buffer = 0
    for train_index, val_index in kf.split(features):
        features_train, features_val = features[train_index], features[val_index]
        labels_train, labels_val = labels[train_index], labels[val_index]
        features = dataframe.drop(columns=["Survived"]).values
        train_dataset = xgb.DMatrix(features_train, labels_train)
        booster = xgb.train(param, train_dataset, num_round)
        test_dataset = xgb.DMatrix(features_val)
        val_predictions =  booster.predict(test_dataset)
        fpr, tpr, threshold = metrics.roc_curve(labels_val, val_predictions)
        roc_auc = metrics.auc(fpr, tpr)
        auc_buffer += roc_auc
    return auc_buffer / 5.0

def train_booster(dataframe):
    print(dataframe.columns)
    labels = dataframe.Survived.values
    features = dataframe.drop(columns=["Survived"]).values
    dataset = xgb.DMatrix(features, labels)
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 3
    booster = xgb.train(param, dataset, num_round)
    return booster
