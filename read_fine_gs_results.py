import pandas as pd
from validation import read_grid_search_results
import ast
from scipy.stats import rankdata
import json

results_paths = [
	'fine_gs_results_giulia1.csv',
	'fine_gs_results_giulia2.csv',
	'fine_gs_results_giulia3.csv',
	'fine_gs_results_giulia4.csv',
	'fine_gs_results_irene1.csv',
	'fine_gs_results_irene2.csv',
	'fine_gs_results_irene3.csv',
	'fine_gs_results_irene4.csv'
]
all_results_path = 'fine_gs.csv'
ranked_results_path = 'fine_gs_params_rank.csv'
K = 5

# concatenate results into a single dataframe
scores_df = pd.DataFrame(columns=[])
for path in results_paths:
	partial_scores_df = pd.read_csv(path, sep=",")
	scores_df = pd.concat([scores_df, partial_scores_df], ignore_index=True)

# rank results
scores_df['val_mee_mean_rank'] = rankdata(scores_df['val_mee_mean'], method='dense')
scores_df['val_mse_mean_rank'] = rankdata(scores_df['val_mse_mean'], method='dense')
scores_df['tr_mee_mean_rank'] = rankdata(scores_df['tr_mee_mean'], method='dense')
scores_df['tr_mse_mean_rank'] = rankdata(scores_df['tr_mse_mean'], method='dense')

# sort results by 'val_score_mean_rank'
scores_df = scores_df.sort_values(by=['val_mee_mean_rank'], ignore_index=True)
scores_df.drop(scores_df.columns[0], axis=1, inplace=True)

columns_order = [
	'val_mse_mean_rank',
	'val_mse_mean',
	'val_mse_dev',
	'val_mee_mean_rank',
	'val_mee_mean',
	'val_mee_dev',
	'tr_mse_mean_rank',
	'tr_mse_mean',
	'tr_mse_dev',
	'tr_mee_mean_rank',
	'tr_mee_mean',
	'tr_mee_dev',
]
for i in range(K):
	columns_order.append('split%d_val_mse'%(i))
	columns_order.append('split%d_val_mee'%(i))
	columns_order.append('split%d_tr_mse'%(i))
	columns_order.append('split%d_tr_mee'%(i))
	columns_order.append('split%d_best_epoch'%(i))
columns_order.append('params')
scores_df = scores_df[columns_order]

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
for param in scores_df['params']: # TODO: fare in modo migliore?
	params_dict = json.loads(param)
	#params_dict = ast.literal_eval(param)
	for key in rem_list:
		del params_dict[key]
		params_dict['batch_size'] = str(params_dict['batch_size'])
	params_df = pd.concat([params_df, pd.DataFrame([params_dict])], ignore_index=True)
params_df.to_csv(ranked_results_path)

# write all results into a single csv
scores_df.to_csv(all_results_path)
