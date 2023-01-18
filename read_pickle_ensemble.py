import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils import mee, mse


def pad(a):
    l = np.array([len(a[i]) for i in range(len(a))])
    width = l.max()
    b=[]
    for i in range(len(a)):
        if len(a[i]) != width:
            x = np.pad(a[i], (0,width-len(a[i])), 'constant',constant_values = 0)
        else:
            x = a[i]
        b.append(x)
    b = np.array(b)
    return b


file = open('./ensemble_assesment.pkl', 'rb')
data = pickle.load(file)
file.close()


train_losses = np.array(data['train_losses'])
train_scores = np.array(data['train_scores'])
val_losses = np.array(data['val_losses'])
val_scores = np.array(data['val_scores'])
epochs = np.array(data['best epochs'])

# loss per ogni modello
for i in range(5):
    plt.figure()
    for j in range(5):
        if j == 0:
            plt.semilogy(train_losses[j], label='Development set (MSE)', color='blue', linewidth=0.8)
            plt.semilogy(val_losses[j], label='Internal test set (MSE)', color='red', linewidth=0.8)
        else:
            plt.semilogy(train_losses[j], color='blue', linewidth=0.8)
            plt.semilogy(val_losses[j], color='red', linewidth=0.8)
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Log(Loss)')
    plt.ylim(top=10)
    plt.savefig('mse_curves_model%d.pdf' % i, bbox_inches="tight")

# score per ogni modello
plt.figure()
for i in range(5):
    plt.figure()
    for j in range(5):
        if j == 0:
            plt.semilogy(train_scores[j], label='Development set (MEE)', color='blue', linewidth=0.6)
            plt.semilogy(val_scores[j], label='Internal test set (MEE)', color='red', linewidth=0.6)
        else:
            plt.semilogy(train_scores[j], color='blue', linewidth=0.6)
            plt.semilogy(val_scores[j], color='red', linewidth=0.6)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Log(Error)')
    plt.ylim(top=5)
    plt.savefig('mee_curves_model%d.pdf' % i, bbox_inches="tight")

# loss di tutti i modelli
plt.figure()
for i in range(50):
    if i==49:
            plt.semilogy(train_losses[i], label='Development set (MSE)', color='lightsteelblue', linewidth=0.6)
            plt.semilogy(val_losses[i], label='Internal test set (MSE)', color='lightcoral', linewidth=0.6)
    else:
            plt.semilogy(train_losses[i], color='lightsteelblue', linewidth=0.6)
            plt.semilogy(val_losses[i], color='lightcoral', linewidth=0.6)

train_losses = pad(train_losses)
val_losses = pad(val_losses)
train_losses_mean = np.average(train_losses, weights=(train_losses > 0), axis=0)
val_losses_mean = np.average(val_losses, weights=(train_losses > 0), axis=0)

plt.semilogy(train_losses_mean, color='blue', label='Ensemble development set (MSE)')
plt.semilogy(val_losses_mean , color='red', linestyle='-', label='Ensemble internal test set (MSE)')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Loss)')
plt.ylim(top=10)
plt.savefig('mse_curves.pdf', bbox_inches="tight")

# score di tutti i modelli
plt.figure() 
for i in range(50):
    if i==49:
            plt.semilogy(train_scores[i], label='Development set (MEE)', color='lightsteelblue', linewidth=0.5)
            plt.semilogy(val_scores[i], label='Internal test set (MEE)', color='lightcoral', linewidth=0.5)
    else:
            plt.semilogy(train_scores[i], color='lightsteelblue', linewidth=0.5)
            plt.semilogy(val_scores[i], color='lightcoral', linewidth=0.5)

train_scores = pad(train_scores)
val_scores = pad(val_scores)
train_scores_mean = np.average(train_scores, weights=(train_scores > 0), axis=0)
val_scores_mean = np.average(val_scores, weights=(val_losses > 0), axis=0)
plt.semilogy(train_scores_mean, color='blue', label='Ensemble development set (MEE)')
plt.semilogy(val_scores_mean, color='red', linestyle='-', label='Ensemble internal test set (MEE)')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Error)')
plt.ylim(top=5)
plt.savefig('mee_curves.pdf', bbox_inches="tight")

# loss media
plt.figure()
plt.semilogy(train_losses_mean, color='blue', label='Ensemble development set (MSE)')
plt.semilogy(val_losses_mean , 'r--', linestyle='-', label='Ensemble internal test set (MSE)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Loss)')
plt.savefig('mse_mean_curves.pdf', bbox_inches="tight")

#score medio
plt.figure()
plt.semilogy(train_scores_mean, color='blue', label='Ensemble development set (MEE)')
plt.semilogy(val_scores_mean, 'r--', linestyle='-', label='Ensemble internal test set (MEE)')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log(Error)')
plt.savefig('mee_mean_curves.pdf', bbox_inches="tight")


# dataframe

train_loss_mean = 0
train_score_mean = 0
val_loss_mean = 0
val_score_mean = 0

n_models = 50

for i in range(n_models):
    train_loss_mean += train_losses[i][epochs[i]]
    train_score_mean += train_scores[i][epochs[i]]
    val_loss_mean += val_losses[i][epochs[i]]
    val_score_mean += val_scores[i][epochs[i]]

train_loss_mean /= n_models
train_score_mean /= n_models
val_loss_mean /= n_models
val_score_mean /= n_models


preds = data['preds']
y_test = data['true']
mean_preds = np.mean(np.array(preds), axis=0)
mse_test = mse(mean_preds, y_test)
mee_test = mee(mean_preds, y_test)

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
# TODO: predizioni di ogni modello non le salviamo

preds_path = 'preds_assesment.csv'
df_preds = pd.DataFrame(mean_preds)
df_preds.to_csv(preds_path)