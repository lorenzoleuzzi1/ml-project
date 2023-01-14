import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv
from cup_parsing import load_dev_set_cup


grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[30, 30], [30, 60], [60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [200],
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.0001, 0.00025, 0.0005, 0.001],
        'alpha': [0.75, 0.8, 0.85, 0.9],
        'verbose': [False],
        'nesterov' : [True, False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True], 
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init': [0.001, 0.0005, 0.0001]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[30, 30], [30, 60], [60, 30], [30, 30, 30]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [200, 500],
        'tol' : [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001, 0.00025, 0.0005, 0.001],
        'alpha': [0.75, 0.8, 0.85, 0.9],
        'verbose': [False],
        'nesterov' : [True, False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init':[0.2, 0.1, 0.01, 0.005]
    }
])

print(len(grid))
grid_splitted = np.array_split(grid, 9)

print(grid_splitted[4][8])

print(grid_splitted[5][3])

print(grid_splitted[6][0])

print(grid_splitted[7][9])
print(grid_splitted[7][10])
print(grid_splitted[7][25])
print(grid_splitted[7][26])
print(grid_splitted[7][41])
print(grid_splitted[7][42])
print(grid_splitted[7][57])
print(grid_splitted[7][58])
print(grid_splitted[7][73])
print(grid_splitted[7][74])
print(grid_splitted[7][89])
print(grid_splitted[7][90])
print(grid_splitted[7][105])
print(grid_splitted[7][106])
print(grid_splitted[7][107])

"""X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=5, results_path=results_path, evaluation_metric='mse')"""