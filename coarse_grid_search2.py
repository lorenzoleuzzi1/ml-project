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

grid = ParameterGrid(
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[20, 20], [60], [20, 20, 20]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [500],
        'tau' : [200], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.0001, 0.001, 0.0005],
        'alpha': [0.7, 0.9],
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
        'learning_rate_init': [0.0005, 0.005, 0.05] # 0.0005 perchè quando alpha=0 e batch size 64 lr deve essere più piccolo
    }
)

print(len(grid))
grid_splitted = np.array_split(grid, 3)

#results_path = 'coarse_gs2_results_giulia.csv'
#grid = grid_splitted[0]

#results_path = 'coarse_gs2_results_irene.csv'
#grid = grid_splitted[1]

results_path = 'coarse_gs2_results_lorenzo.csv'
grid = grid_splitted[2]

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=3, results_path=results_path, evaluation_metric='mse')