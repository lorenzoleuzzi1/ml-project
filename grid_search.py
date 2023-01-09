from sklearn.model_selection import ParameterGrid
from network import Network
import pandas as pd
from scipy.stats import rankdata

grid = ParameterGrid([
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
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
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
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
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [256, 1.0],
        'learning_rate_init': [0.0005, 0.001, 0.1]
    },
    {
        'activation_out': ['identity'],
        'classification' : [False],
        'activation_hidden': ['tanh', 'relu'],
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
        'stopping_patience': [5],
        'validation_size': [0.1],
        'validation_frequency': [5],
        'random_state': [None],
        'reinit_weights': [True],
        'weights_dist': [None],
        'weights_bound': [None],
        'metric_decrease_tol': [0.1/100],
        'batch_size': [1],
        'learning_rate_init': [0.0001, 0.0005, 0.001]
    }
    ]
)

print(len(grid))

scores = {}
scores['tr_loss_mean'] = 1
scores['tr_loss_dev'] = 2
scores['val_score_mean'] = 3
scores['val_score_dev'] = 4

df_params =  pd.DataFrame(columns=[])
df_scores =  pd.DataFrame(columns=[])
params_list = []
for params in grid:
    params_list.append(params)
    df_params = pd.concat([df_params, pd.DataFrame([params])], ignore_index=True)
    df_scores = pd.concat([df_scores, pd.DataFrame([scores])], ignore_index=True)
    df_scores['tr_loss_rel_dev'] = df_scores['tr_loss_dev'] / df_scores['tr_loss_mean']
    df_scores['val_score_rel_dev'] = df_scores['val_score_dev'] / df_scores['val_score_mean']

df_scores['tr_loss_mean_rank'] = rankdata(df_scores['tr_loss_mean'], method='dense')
df_scores['tr_loss_rel_dev'] = rankdata(df_scores['tr_loss_rel_dev'], method='dense')
df_scores['val_score_mean_rank'] = rankdata(df_scores['val_score_mean'], method='dense')
df_scores['val_score_rel_dev'] = rankdata(df_scores['val_score_rel_dev'], method='dense')
df_scores['params'] = params_list

df_scores.to_csv('scores_df.csv')
df_params.drop(['classification', 'verbose'], axis=1) # drop also random state, reinit weights... (?)
df_params.to_csv('params_df.csv')

# nomi colonne migliori
# itegra con validation
# salva anche split values

# CSV --> PANDAS DATAFRAME --> DICT
read_df = pd.read_csv('scores_df.csv', sep=",")
read_df.drop(read_df.columns[0], axis=1, inplace=True) # drop first column
read_dict = read_df.to_dict(orient='records')
print(read_dict[0]) # print row 0 as dict