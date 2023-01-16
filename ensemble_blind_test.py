import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import Network
from cup_parsing import load_internal_test_cup, read_tr_cup
from validation import read_grid_search_results

df = read_grid_search_results("fine_gs2_results.csv")
X_train, y_train = read_tr_cup()
X_test = load_internal_test_cup()

n_trials = 5
best_n = 10
preds = []
train_losses = []
train_scores = []
epochs = []

for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    print(config)

    for j in range(n_trials):
        net = Network(**config)
        net.fit(X_train, y_train)
        if i==0 and j==0:
            plt.figure("mse")
            plt.semilogy(net.train_losses, label='Development set + internal test set (MSE)', color='lightsteelblue', linewidth=0.5)
        else:
            plt.figure("mse")
            plt.semilogy(net.train_losses, color='lightsteelblue', linewidth=0.5)
        if i==0 and j==0:
            plt.figure("mee")
            plt.semilogy(net.train_scores, label='Development set + internal test set (MEE)', color='lightsteelblue', linewidth=0.5)
        else:
            plt.figure("mee")
            plt.semilogy(net.train_scores, color='lightsteelblue', linewidth=0.5, )
        
        train_losses.append(net.train_losses)
        train_scores.append(net.train_scores)
        Y_pred = net.predict(X_test)
        preds.append(Y_pred)
        epochs.append(net.best_epoch)

train_losses_mean = np.mean(train_losses, axis=0)
train_scores_mean = np.mean(train_scores, axis=0)

train_loss_mean = 0
train_score_mean = 0

n_models = best_n*n_trials

for i in range(n_models):
    train_loss_mean += train_losses[i][epochs[i]]
    train_score_mean += train_scores[i][epochs[i]]

train_loss_mean /= n_models
train_score_mean /= n_models

data = {
    'preds': preds,
    'best epochs': epochs,
    'train_losses': train_losses,
    'train_losses_mean': train_losses_mean,
    'train_scores': train_scores,
    'train_scores_mean': train_scores_mean,
    'train_loss_mean_best_epoch': train_loss_mean,
    'train_score_mean_best_epoch': train_score_mean
}

plt.figure("mse")
plt.semilogy(train_losses_mean, color='blue', label='Development set + internal test set (MSE)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Loss)')
plt.savefig('mse_curves.pdf', bbox_inches="tight")

plt.figure("mee")
plt.semilogy(train_scores_mean, color='blue', label='Development set + internal test set (MEE)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Error)')
plt.savefig('mee_curves.pdf', bbox_inches="tight")

mean_preds = np.mean(np.array(preds), axis=0)
data['mean_preds'] = mean_preds

file = open('./ensemble_blind_test.pkl', 'wb')
pickle.dump(data, file)
file.close()

csv_data = {}
csv_data['train_loss_mean_best_epoch'] = train_loss_mean
csv_data['train_score_mean_best_epoch'] = train_score_mean

results_path = './ensemble_blind_test.csv'
df_scores = pd.DataFrame([csv_data, csv_data])
df_scores.to_csv(results_path)

df = pd.DataFrame(mean_preds)
f = open('team-name_ML-CUP22-TS.csv', 'a')
f.write('# Giulia Ghisolfi	Lorenzo Leuzzi	Irene Testa\n')
f.write('# team-name\n')
f.write('# ML-CUP22\n')
f.write('# 23/01/2023\n')
df.to_csv(f, header=False)
f.close()