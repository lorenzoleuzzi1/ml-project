from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup, load_internal_test_cup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import pickle
import numpy as np
from utils import mee


df = read_grid_search_results("fine_gs2_results.csv")
X_train, y_train = load_dev_set_cup()
X_test, y_test = load_internal_test_cup()
n_trials = 5
best_n = 10
preds = []
train_losses = []
val_losses = []
train_scores = []
val_scores = []
epochs = []

fig1, _ = plt.figure() # mse
fig2, _ = plt.figure() # mee

for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    print(config)

    for j in range(n_trials):
        net = Network(**config)
        net.fit(X_train, y_train, X_test, y_test)
        if i==0 and j==0:
            fig1.plot(net.train_losses_reg, label='Development set', color='blue', alpha=0.7, linewidth=0.5)
            fig1.plot(net.val_losses, label='Internal test set', color='red', alpha=0.7, linewidth=0.5)
        else:
            fig1.plot(net.train_losses_reg, color='blue', alpha=0.7, linewidth=0.5)
            fig1.plot(net.val_losses, color='red', alpha=0.7, linewidth=0.5)
        if i==0 and j==0:
            fig2.plot(net.train_scores, label='Development set', color='blue', alpha=0.7, linewidth=0.5)
            fig2.plot(net.val_scores, label='Internal test set', color='red', alpha=0.7, linewidth=0.5)
        else:
            fig2.plot(net.train_scores, color='blue', alpha=0.7, linewidth=0.5)
            fig2.plot(net.val_scores, color='red', alpha=0.7, linewidth=0.5)
        
        train_losses.append(net.train_losses_reg)
        val_losses.append(net.val_losses)
        train_scores.append(net.train_scores)
        val_scores.append(net.val_scores)
        Y_pred = net.predict(y_test)
        preds.append(Y_pred)
        epochs.append(net.best_epochs)
    
data = {
    'preds': preds,
    'true': y_test,
    'best epochs': epochs,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_scores': train_scores,
    'val_scores': val_scores
}

fig1.plot(np.mean(train_losses, axis=0), color='blue', label='Ensemble development set')
fig1.plot(np.mean(val_losses, axis=0), color='red', linestyle='-', label='Ensemble internal test set')
fig2.plot(np.mean(train_scores, axis=0), color='blue', label='Ensemble development set')
fig2.plot(np.mean(val_scores, axis=0), color='red', linestyle='-', label='Ensemble internal test set')
fig1.legend()
fig1.grid()
fig1.xlabel('Epochs')
fig1.ylabel('Loss (MSE)')

fig2.legend()
fig2.grid()
fig1.xlabel('Epochs')
fig1.ylabel('Error (MEE)')

fig1.savefig('')

file = open('/kaggle/working/ensemble_kfold.pkl', 'wb')
pickle.dump(data, file)
file.close()

mean_preds = np.mean(np.array(preds), axis=0)
mee()

results_path = '/kaggle/working/ensemble_k_fold.csv'
df_scores = pd.DataFrame([diz, diz])
df_scores.to_csv(results_path)