import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv
from cup_parsing import load_dev_set_cup


grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[20, 20], [30, 30], [60, 30], [30, 60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [200], # after tau epochs lr=1/10*init
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
        'learning_rate_init': [0.001, 0.0005, 0.0001] # 0.0005 perchè quando alpha=0 e batch size 64 lr deve essere più piccolo
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[20, 20], [30, 30], [60, 30], [30, 60]],
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['linear_decay'],
        'lambd': [0.0001, 0.00025, 0.0005, 0.001],
        'alpha': [0.75, 0.8, 0.85, 0.9],
        'verbose': [False],
        'nesterov' : [True,False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init':[0.1, 0.05, 0.01, 0.005]
    }
])

print(len(grid))
grid_splitted = np.array_split(grid, 3)

results_path = 'fine_gs_results_giulia.csv'
grid = grid_splitted[0]

#results_path = 'fine_gs_results_irene.csv'
#grid = grid_splitted[1]

#results_path = 'fine_gs_results_lorenzo.csv'
#grid = grid_splitted[2]

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=3, results_path=results_path, evaluation_metric='mse')