import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv
from cup_parsing import load_dev_set_cup


grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [ [60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.001, 0.0005, 0.00025, 0.0001, 0.00005],
        'alpha': [0.7, 0.75, 0.8, 0.85, 0.9],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init': [0.01, 0.008, 0.006, 0.004]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [ [60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.001, 0.0005, 0.00025, 0.0001, 0.00005],
        'alpha': [0.75, 0.8, 0.85, 0.9],
        'verbose': [False],
        'nesterov' : [True],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init': [0.008, 0.006, 0.004, 0.002]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [ [60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.001, 0.0005, 0.00025, 0.0001, 0.00005],
        'alpha': [0.7],
        'verbose': [False],
        'nesterov' : [True],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init': [0.01, 0.008, 0.006, 0.004]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [300],
        'tol' : [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.005, 0.001, 0.0005, 0.00025, 0.0001],
        'alpha': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
        'verbose': [False],
        'nesterov' : [False, True],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init':[0.12, 0.1, 0.08, 0.06, 0.05, 0.04]
    }
])

print(len(grid))

grid_splitted = np.array_split(grid, 9)


"""results_path = '/kaggle/working/fine_gs2_results_giulia1.csv'
grid = grid_splitted[0]

results_path = '/kaggle/working/fine_gs2_results_giulia2.csv'
grid = grid_splitted[1]

results_path = '/kaggle/working/fine_gs2_results_giulia3.csv'
grid = grid_splitted[2]

results_path = '/kaggle/working/fine_gs2_results_irene1.csv'
grid = grid_splitted[3]

results_path = '/kaggle/working/fine_gs2_results_irene2.csv'
grid = grid_splitted[4]

results_path = '/kaggle/working/fine_gs2_results_irene3.csv'
grid = grid_splitted[5]

results_path = '/kaggle/working/fine_gs2_results_lorenzo1.csv'
grid = grid_splitted[6]

results_path = '/kaggle/working/fine_gs2_results_lorenzo2.csv'
grid = grid_splitted[7]

results_path = '/kaggle/working/fine_gs2_results_lorenzo3.csv'
grid = grid_splitted[8]"""

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=5, results_path=results_path, evaluation_metric='mse')