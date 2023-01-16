from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup, load_internal_test_cup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import pickle
import numpy as np
from utils import mee, mse


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
            fig1.semilogy(net.train_losses, label='Development set (MSE)', color='blue', alpha=0.7, linewidth=0.5)
            fig1.semilogy(net.val_losses, label='Internal test set (MSE)', color='red', alpha=0.7, linewidth=0.5)
        else:
            fig1.semilogy(net.train_losses, color='blue', alpha=0.7, linewidth=0.5)
            fig1.semilogy(net.val_losses, color='red', alpha=0.7, linewidth=0.5)
        if i==0 and j==0:
            fig2.semilogy(net.train_scores, label='Development set (MEE)', color='blue', alpha=0.7, linewidth=0.5)
            fig2.semilogy(net.val_scores, label='Internal test set (MEE)', color='red', alpha=0.7, linewidth=0.5)
        else:
            fig2.semilogy(net.train_scores, color='blue', alpha=0.7, linewidth=0.5)
            fig2.semilogy(net.val_scores, color='red', alpha=0.7, linewidth=0.5)
        
        train_losses.append(net.train_losses)
        val_losses.append(net.val_losses)
        train_scores.append(net.train_scores)
        val_scores.append(net.val_scores)
        Y_pred = net.predict(X_test)
        preds.append(Y_pred)
        epochs.append(net.best_epochs)

train_losses_mean = np.mean(train_losses, axis=0)
val_losses_mean = np.mean(val_losses, axis=0)
train_scores_mean = np.mean(train_scores, axis=0)
val_scores_mean = np.mean(val_scores, axis=0)

train_loss_mean = 0
train_score_mean = 0
val_loss_mean = 0
val_score_mean = 0

n_models = best_n*n_trials

for i in range(n_models):
    train_loss_mean += train_losses[i][epochs[i]]
    train_score_mean += train_scores[i][epochs[i]]
    val_loss_mean += val_losses[i][epochs[i]]
    val_score_mean += val_scores[i][epochs[i]]

train_loss_mean /= n_models
train_score_mean /= n_models
val_loss_mean /= n_models
val_score_mean /= n_models

data = {
    'preds': preds,
    'true': y_test,
    'best epochs': epochs,
    'train_losses': train_losses,
    'train_losses_mean': train_losses_mean,
    'train_scores': train_scores,
    'train_scores_mean': train_scores_mean,
    'val_losses': val_losses,
    'val_losses_mean': val_losses_mean,
    'val_scores': val_scores,
    'val_scores_mean': val_scores_mean,
    'train_loss_mean_best_epoch': train_loss_mean,
    'train_score_mean_best_epoch': train_score_mean,
    'val_loss_mean_best_epoch': val_loss_mean,
    'val_score_mean_best_epoch': val_score_mean
}

fig1.semilogy(train_losses_mean, color='blue', label='Ensemble development set (MSE)')
fig1.semilogy(val_losses_mean , color='red', linestyle='-', label='Ensemble internal test set (MSE)')
fig2.semilogy(train_scores_mean, color='blue', label='Ensemble development set (MEE)')
fig2.semilogy(val_scores_mean, color='red', linestyle='-', label='Ensemble internal test set (MEE)')
fig1.legend()
fig1.grid()
fig1.xlabel('Epochs')
fig1.ylabel('Log(Loss)')

fig2.legend()
fig2.grid()
fig1.xlabel('Epochs')
fig1.ylabel('Log(Error)')

fig1.savefig('mse_curves.pdf', bbox_inches="tight")
fig1.savefig('mee_curves.pdf', bbox_inches="tight")

mean_preds = np.mean(np.array(preds), axis=0)
mse_test = mse(mean_preds, y_test)
mee_test = mee(mean_preds, y_test)

data['mse_test'] = mse_test
data['mee_test'] = mee_test
data['mean_preds'] = mean_preds

file = open('/ensemble_assesment.pkl', 'wb')
pickle.dump(data, file)
file.close()

csv_data = {}
csv_data['mse_test'] = mse_test
csv_data['mee_test'] = mee_test
csv_data['train_loss_mean_best_epoch'] = train_loss_mean
csv_data['train_score_mean_best_epoch'] = train_score_mean
csv_data['val_loss_mean_best_epoch'] = val_loss_mean
csv_data['val_score_mean_best_epoch'] = val_score_mean

results_path = 'ensemble_assesment.csv'
df_scores = pd.DataFrame([csv_data, csv_data])
df_scores.to_csv(results_path)