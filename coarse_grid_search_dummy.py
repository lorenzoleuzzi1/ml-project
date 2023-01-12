import numpy as np
from sklearn.model_selection import ParameterGrid
from validation import grid_search_cv, read_grid_search_results
from cup_parsing import load_dev_set_cup

grid = ParameterGrid(
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['logistic'],
        'hidden_layer_sizes': [[20, 20]], 
        'loss': ['mse'],
        'evaluation_metric' : ['mee'],
        'epochs': [10],
        'tau' : [10], # after tau epochs lr=1/10*init
        'tol' : [0.0001],
        'learning_rate': ['fixed'],
        'lambd': [0.0005],
        'alpha': [0.9],
        'verbose': [False],
        'nesterov' : [False],
        'early_stopping': [False],
        'stopping_patience': [9],
        'validation_size': [0.1],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.00001],
        'batch_size': [128],
        'learning_rate_init': [0.0005, 0.005] # 0.0005 perchè quando alpha=0 e batch size 64 lr deve essere più piccolo
    }
)

print(len(grid))

X_dev, y_dev = load_dev_set_cup()
grid_search_cv(grid=grid, X=X_dev, y=y_dev, k=3, results_path="dummy.csv", evaluation_metric='mse')
df = read_grid_search_results("dummy.csv")
print(df)
