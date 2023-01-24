from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def auc(ytest, ypred):
    auc = roc_auc_score(ytest, ypred)
    return auc

