import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from validation import k_fold_cross_validation, grid_search_cv
from network import Network
from utils import error_plot, accuracy_plot
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import json

MONKS1_TRAIN_PATH = './datasets/monks-1.train'
MONKS1_TEST_PATH = './datasets/monks-1.test'
MONKS2_TRAIN_PATH = './datasets/monks-2.train'
MONKS2_TEST_PATH = './datasets/monks-2.test'
MONKS3_TRAIN_PATH = './datasets/monks-3.train'
MONKS3_TEST_PATH = './datasets/monks-3.test'

def read_monks(path, one_hot_encoding=True, target_rescaling=True):
    data = pd.read_csv(path, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() # int 64
    if one_hot_encoding:
        data = OneHotEncoder().fit_transform(data).toarray() # float 64
    if target_rescaling:
        targets[targets == 0] = -1 # int 64
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets


grid3 = ParameterGrid([
    {
        'activation_out': ['tanh', 'logistic'],
        'classification' : [True],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[4]],
        'loss': ['mse'],
        'evaluation_metric' : ['accuracy'],
        'epochs': [200],
        'tol' : [0.000001],
        'learning_rate': ['fixed'],
        'lambd': [0],
        'alpha': [0.9],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': ['uniform'],
        'weights_bound': [0.7],
        'metric_decrease_tol': [0.000001],
        'batch_size': [1, 8, 16, 32],
        'learning_rate_init': [0.05, 0.01, 0.005, 0.001, 0.0001],
        'stopping_criteria_on_loss': [True]
    },
    {
        'activation_out': ['tanh', 'logistic'],
        'classification' : [True],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[4]],
        'loss': ['mse'],
        'evaluation_metric' : ['accuracy'],
        'epochs': [200],
        'tol' : [0.000001],
        'learning_rate': ['fixed'],
        'lambd': [0],
        'alpha': [0.9],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.000001],
        'batch_size': [1, 8, 16, 32],
        'learning_rate_init': [0.05, 0.01, 0.005, 0.001, 0.0001],
        'stopping_criteria_on_loss': [True]
    }]
)

X_train3, y_train3 = read_monks(MONKS3_TRAIN_PATH)
X_test3, y_test3 = read_monks(MONKS3_TEST_PATH)

n_trial = 5
for param in grid3:
    print(f"starting grid search - exploring {len(grid3)} configs")
    df_scores = pd.DataFrame(columns=[])
    for i, config in enumerate(grid3):
        print(f"{i+1}/{len(grid3)}")
        network = Network(**config)
        dict_row = {}
        train_losses = 0
        train_scores = 0
        val_losses = 0
        val_scores = 0
        accuracy_mean = 0
        for i in range(n_trial):
            network.fit(X_train3, y_train3, X_test3, y_test3)
            accuracy = network.score(X_test3, y_test3, 'accuracy')
            dict_row['trial%d_train_loss'%i] = network.train_losses[network.best_epoch]
            dict_row['trial%d_train_score'%i] = network.train_scores[network.best_epoch]
            dict_row['trial%d_val_loss'%i] = network.val_losses[network.best_epoch]
            dict_row['trial%d_val_score'%i] = network.val_scores[network.best_epoch]
            dict_row['trial%d_best_epoch'%i] = network.best_epoch
            dict_row['trial%d_accuracy'%i] = accuracy
            train_losses += network.train_losses[network.best_epoch]
            train_scores += network.train_scores[network.best_epoch]
            val_losses += network.val_losses[network.best_epoch]
            val_scores += network.val_scores[network.best_epoch]
            accuracy_mean += accuracy

        train_losses /= n_trial
        train_scores /= n_trial
        val_losses /= n_trial
        val_scores /= n_trial
        accuracy_mean /= n_trial
        dict_row['mean_accuracy'] = accuracy_mean
        dict_row['mean_train_loss'] = train_losses
        dict_row['mean_train_score'] = train_scores
        dict_row['mean_val_loss'] = val_losses
        dict_row['val_scores'] = val_scores
        dict_row['params'] = json.dumps(config)
        df_scores = pd.concat([df_scores, pd.DataFrame([dict_row])], ignore_index=True)

df_scores.to_csv('/kaggle/working/monks3_gs.csv')