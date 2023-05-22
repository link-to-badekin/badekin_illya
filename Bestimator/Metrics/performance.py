#from sklearn.metrics import roc_curve
import torch
import numpy as np

def Equal_Error_Rate (y_true, y_pred):
    pass
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer