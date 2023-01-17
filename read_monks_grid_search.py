import pandas as pd
from scipy.stats import rankdata
import json


results_path = 'monks1_gs.py'
all_results_path = 'monks1_results.csv'
ranked_results_path = 'monks1_params_rank.csv'

# concatenate results into a single dataframe
scores_df = pd.DataFrame(columns=[])
scores_df = pd.read_csv(results_path, sep=",")

# rank results
scores_df['mean_train_loss_rank'] = rankdata(scores_df['mean_train_loss'], method='dense')
scores_df['mean_train_score_rank'] = rankdata(scores_df['mean_train_score'], method='dense')
scores_df['mean_val_loss_rank'] = rankdata(scores_df['mean_val_loss'], method='dense')
scores_df['mean_accuracy_rank'] = rankdata(scores_df['mean_accuracy'], method='dense')

# sort results by 'val_score_mean_rank'
scores_df = scores_df.sort_values(by=['mean_accuracy_rank'], ignore_index=True)
scores_df.drop(scores_df.columns[0], axis=1, inplace=True)

columns_order = [
	'mean_accuracy_rank',
	'mean_accuracy',
	'mean_val_loss_rank',
	'mean_val_loss',
	'mean_train_score_rank',
	'mean_train_score',
	'mean_train_loss_rank',
	'mean_train_loss'
	]

trial = 5
for i in range(trial):
	columns_order.append('trial%d_val_score'%i)
	columns_order.append('trial%d_val_loss'%i)
	columns_order.append('trial%d_accuracy'%i)
	columns_order.append('trial%d_train_score'%i)
	columns_order.append('trial%d_train_loss'%i)
	columns_order.append('trial%d_best_epoch'%i)
columns_order.append('params')
scores_df = scores_df[columns_order]

# write params as csv deleting fixed params
rem_list = [ 
	'classification',
	'loss',
	'evaluation_metric',
	'epochs',
	'tol',
	'learning_rate',
	'alpha',
	'verbose',
	'nesterov',
	'early_stopping',
	'stopping_patience',
	'random_state',
	'reinit_weights',
	'metric_decrease_tol',
	'stopping_criteria_on_loss'
	]

params_df = pd.DataFrame(columns=[])
for param in scores_df['params']: # TODO: fare in modo migliore?
	params_dict = json.loads(param)
	#params_dict = ast.literal_eval(param)
	for key in rem_list:
		if key in params_dict:
			del params_dict[key]
	params_df = pd.concat([params_df, pd.DataFrame([params_dict])], ignore_index=True)
params_df.to_csv(ranked_results_path)

# write all results into a single csv
scores_df.to_csv(all_results_path)
