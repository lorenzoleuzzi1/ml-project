import pandas as pd
from validation import read_grid_search_results
import ast
from scipy.stats import rankdata
import json

dataframe = read_grid_search_results('fine_gs_edited.csv')

# write params as csv deleting fixed params
rem_list = [ 
	'activation_out',
	'classification',
	'early_stopping',
	'evaluation_metric',
	'loss',
	'metric_decrease_tol',
	'random_state',
	'reinit_weights',
	'stopping_patience',
	'tol',
	'validation_size',
	'verbose',
	'weights_bound',
	'weights_dist',
	'epochs'
	]
params_df = pd.DataFrame(columns=[])
for i, param in enumerate(dataframe['params']):
	for key in rem_list:
		del param[key]
	if dataframe['overflow'][i]==True:
		params_df = pd.concat([params_df, pd.DataFrame([param])], ignore_index=True)
params_df.to_csv('fine_gs_overflow_param.csv')
