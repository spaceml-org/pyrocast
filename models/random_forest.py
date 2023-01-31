from sklearn.ensemble import RandomForestClassifier
import utils.dataprep as dp
from utils import metrics
import numpy as np
import joblib


class RandomForest(RandomForestClassifier):
    """ SciKit Learn model initalised with paper hyperparameters"""
    def __init__(self, n_estimators=500, max_depth=10, class_weight="balanced_subsample", random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state


if __name__ in "___main__":
    # Load datacubes (you can dowload these from the Pyrocast database)
    geo_cubes = np.load('')
    env_cubes = np.load('')
    flag_cubes = np.load('')
    xtrain, ytrain, xtest, ytest = dp.load_rfdata()
    rf = RandomForest()
    rf.fit(xtrain, ytrain)
    joblib.dump(rf, 'models/instances/test_rf.joblib')
    ypred = rf.predict()
    auc = metrics.auc(ytest, ypred)
    print(auc)
