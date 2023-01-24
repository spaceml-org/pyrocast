from utils.dataprep as dp
from utils import metrics
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class CNN():
    def ___init__():
    

x_train, ytrain, x_test, y_test = dp.load_sets()
    cnn = CNN ()
    y_pred = 
    auc = metrics.auc(y_test, y_pred)
    print(auc)