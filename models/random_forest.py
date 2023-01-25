from sklearn.ensemble import RandomForestClassifier
from utils.dataprep as dp
from utils import metrics


class RandomForest(RandomForestClassifier):
    """ Sci Kit Learn model initalised with our """
    def __init__(n_estimators = 500, max_depth = 10, class_weight = "balanced_subsample", random_state = 0)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random state


if __name__ in "___main__":
    x_train, ytrain, x_test, y_test = dp.load_rfdata()
    clf = RandomForest()
    clf.fit(x_train, y_train)
    y_pred = clf.predict()
    auc = metrics.auc(y_test, y_pred)
    print(auc)