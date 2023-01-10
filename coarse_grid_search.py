import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv
from cup_parsing import load_dev_set_cup

grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0, 0.8],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed', 'linear_decay'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0.5, 0.8],
        'verbose': [False],
        'nesterov' : [True],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'logistic'],
        'hidden_layer_sizes': [[30], [120], [60,60]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [200, 800],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.00001, 0.0001, 0.001, 0.01],
        'alpha': [0, 0.8],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [30],
        'validation_size': [0.1],
        'validation_frequency': [1],
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

print(len(grid))
grid_splitted = np.array_split(grid, 3)
print(len(grid_splitted[0]))
print(len(grid_splitted[1]))
print(len(grid_splitted[2]))
print("First config giulia")
print(grid_splitted[0][0])
print("First config irene")
print(grid_splitted[1][0])
print("First config lorenzo")
print(grid_splitted[2][0])

"""results_path = 'coarse_gs_results_giulia.csv'
grid = grid_splitted[0]

results_path = 'coarse_gs_results_irene.csv'
grid = grid_splitted[1]

results_path = 'coarse_gs_results_lorenzo.csv'
grid = grid_splitted[2]

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=3, results_path=results_path)"""

# TODO:
# score funziona?
# assicurarsi che shuffle con random seed fissato mescoli davvero i pattern...
# rivedi tutti i parametri
# lambda range doppi?
# rivedi slides validation
# giustifica topologia
# alpha 0.5 con nesterov false?
# weights dist?
# batch size intermedia? (X_dev_size/3)*2 = 795 esempi per train fold, 795/256 = 3.1 batches
# lr stocastico troppo grande? lr batch troppo piccolo?
# 3 fold coarse, fine?
# weights init for the final model? (before assesment, after assesment)