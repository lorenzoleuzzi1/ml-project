import numpy as np
import pickle
import pandas as pd
from utils import mee

file = open('./ensemble_kfold.pkl', 'rb')
data = pickle.load(file)
file.close()

preds = data['preds']
Y_true = data['true']

pred_k1 = []
pred_k2 = []
pred_k3 = []
pred_k4 = []
pred_k5 = []

for pred in preds:
    k = 1
    for pred_kth in pred:
        if k == 1:
            pred_k1.append(pred_kth)
        if k == 2:
            pred_k2.append(pred_kth)
        if k == 3:
            pred_k3.append(pred_kth)
        if k == 4:
            pred_k4.append(pred_kth)
        if k == 5:
            pred_k5.append(pred_kth)
        k += 1

means_pred_k1 = np.mean(pred_k1, axis=0)
means_pred_k2 = np.mean(pred_k2, axis=0)
means_pred_k3 = np.mean(pred_k3, axis=0)
means_pred_k4 = np.mean(pred_k4, axis=0)
means_pred_k5 = np.mean(pred_k5, axis=0)

split0_val_mee = mee(Y_true[0], means_pred_k1)
split1_val_mee = mee(Y_true[1], means_pred_k2)
split2_val_mee = mee(Y_true[2], means_pred_k3)
split3_val_mee = mee(Y_true[3], means_pred_k4)
split4_val_mee = mee(Y_true[4], means_pred_k5)

val_mee = (split0_val_mee + split1_val_mee + split2_val_mee + split3_val_mee + split4_val_mee)/5

diz = {'split0_val_mee': split0_val_mee,
       'split1_val_mee': split1_val_mee,
       'split2_val_mee': split2_val_mee,
       'split3_val_mee': split3_val_mee,
       'split4_val_mee': split4_val_mee,
       'val_mee': val_mee}


results_path = './ensemble_k_fold.csv'
df_scores = pd.DataFrame([diz, diz])
df_scores.to_csv(results_path)