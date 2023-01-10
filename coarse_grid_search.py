import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv
from cup_parsing import load_dev_set_cup

# RANK WITH FIXED BEST VALUES
# -----pazienza 30 -----
# K=3
#  [60, 60][30, 30] [30, 30, 30] [10, 10] [30] [60] [120]
# ----- pazienza 5 ----
# K=3
# [30, 30] [60, 60] [30, 30, 30] [10, 10] [30] [60] [120]
# k=5
# [30, 30] [60, 60] [30, 30, 30] [30] [10, 10] [60] [120]

grid200 = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0, 0.8],
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
        'batch_size': [64, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1] # 0.0005 perchè quando alpha=0 e batch size 64 lr deve essere più piccolo
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0.5, 0.8],
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
        'batch_size': [64, 1.0],
        'learning_rate_init': [0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0, 0.8],
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
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    }
    ]
)

grid800 = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0, 0.8],
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
        'batch_size': [64, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1] # 0.0005 perchè quando alpha=0 e batch size 64 lr deve essere più piccolo
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0.5, 0.8],
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
        'batch_size': [64, 1.0],
        'learning_rate_init': [0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30, 30], [60, 60], [30, 30, 30]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.00001, 0.0001, 0.001],
        'alpha': [0, 0.8],
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
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    }
    ]
)

grid200_splitted = np.array_split(grid200, 3)
grid800_splitted = np.array_split(grid800, 3)

results_path = 'coarse_gs_results_giulia1.csv'
grid = grid200_splitted[0]

#results_path = 'coarse_gs_results_giulia2.csv'
#grid = grid800_splitted[0]

#results_path = 'coarse_gs_results_irene1.csv'
#grid = grid200_splitted[1]

#results_path = 'coarse_gs_results_irene2.csv'
#grid = grid800_splitted[1]

#results_path = 'coarse_gs_results_lorenzo1.csv'
#grid = grid200_splitted[2]

#results_path = 'coarse_gs_results_lorenzo2.csv'
#grid = grid800_splitted[2]

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=3, results_path=results_path, evaluation_metric='mse')