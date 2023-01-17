from validation import read_grid_search_results
from network import Network
#from cup_parsing import load_dev_set_cup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import pickle
import numpy as np
from utils import mee, mse

#TODO: fatta per verificare se fosse meglio di un solo modello
def k_fold_cross_validation_ensemble(network, X_train, y_train, k):
    if k <= 1:
        print('Number of folds k must be more than 1')
        raise ValueError("k must be more than 1")

    if network.classification:
        kf = StratifiedKFold(n_splits=k, shuffle=True) 
    else:
        kf = KFold(n_splits=k, shuffle=False) # NOTE: to compare different models on the same splits
    
    Y_preds = []
    Y_true = []
    best_epochs = []
    i = 1

    train_losses = [] # list of k values
    train_scores = []
    for train_index, validation_index in kf.split(X_train, y_train):
       
        #-----stratified K-fold split-----
        X_train_fold, X_val_fold = X_train[train_index], X_train[validation_index] 
        y_train_fold, y_val_fold = y_train[train_index], y_train[validation_index] 
        

        # --------------fold train--------------
        network.fit(X_train_fold, y_train_fold)

        best_epoch = network.best_epoch
        train_losses.append(network.train_losses[best_epoch])
        train_scores.append(network.train_scores[best_epoch])
        
        # --------------fold validation--------------
        Y_pred = network.predict_outputs(X=X_val_fold)

        #print("{} fold VL score = {}".format(i, score))    

        best_epochs.append(best_epoch)
        Y_preds.append(Y_pred)
        Y_true.append(y_val_fold)
        i+=1

    return Y_preds, Y_true, best_epochs, train_losses, train_scores



df = read_grid_search_results("fine_gs2_results.csv")
X_train, y_train = load_dev_set_cup()
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#print(df)
#l = np.array([31, 28, 35])
best_n = 10
preds = []
epochs = []
train_losses = []
train_scores = []
for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    print(config)

    net = Network(**config)
    Y_preds, Y_true, best_epochs, train_losses_model, train_scores_model = k_fold_cross_validation_ensemble(net, X_train, y_train, k=5)
    train_losses.append(train_losses_model)
    train_scores.append(train_scores_model)
    preds.append(Y_preds)
    epochs.append(best_epochs)
    
data ={'preds': preds, 'true': Y_true, 'best epochs': epochs, 'train_losses': train_losses, 'train_scores': train_scores}

file = open('/kaggle/working/ensemble_kfold.pkl', 'wb')
pickle.dump(data, file)
file.close()

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

split0_val_mse = mse(Y_true[0], means_pred_k1)
split1_val_mse = mse(Y_true[1], means_pred_k2)
split2_val_mse = mse(Y_true[2], means_pred_k3)
split3_val_mse = mse(Y_true[3], means_pred_k4)
split4_val_mse = mse(Y_true[4], means_pred_k5)

val_mee = (split0_val_mee + split1_val_mee + split2_val_mee + split3_val_mee + split4_val_mee)/5
val_mse = (split0_val_mse + split1_val_mse + split2_val_mse + split3_val_mse + split4_val_mse)/5

train_loss_mean = np.mean(train_losses)
train_score_mean = np.mean(train_scores)

train_loss_splitted = np.mean(train_losses, axis=0)
train_score_splitted = np.mean(train_scores, axis=0)

diz = {'split0_val_mee': split0_val_mee,
       'split1_val_mee': split1_val_mee,
       'split2_val_mee': split2_val_mee,
       'split3_val_mee': split3_val_mee,
       'split4_val_mee': split4_val_mee,
       'val_mee': val_mee,
       'split0_val_mse': split0_val_mse,
       'split1_val_mse': split1_val_mse,
       'split2_val_mse': split2_val_mse,
       'split3_val_mse': split3_val_mse,
       'split4_val_mse': split4_val_mse,
       'val_mse': val_mse,
       'split0_tr_mee': train_score_splitted[0],
       'split1_tr_mee': train_score_splitted[1],
       'split2_tr_mee': train_score_splitted[2],
       'split3_tr_mee': train_score_splitted[3],
       'split4_tr_mee': train_score_splitted[4],
       'tr_mse': train_score_mean,
       'split0_tr_mse': train_loss_splitted[0],
       'split1_tr_mse': train_loss_splitted[1],
       'split2_tr_mse': train_loss_splitted[2],
       'split3_tr_mse': train_loss_splitted[3],
       'split4_tr_mse': train_loss_splitted[4],
       'tr_mse': train_loss_mean
       }

print(diz)

results_path = '/kaggle/working/ensemble_k_fold.csv'
df_scores = pd.DataFrame([diz, diz])
df_scores.to_csv(results_path)