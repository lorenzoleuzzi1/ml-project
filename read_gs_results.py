import pandas as pd
from scipy.stats import rankdata

# TODO: togliere gli indici di riga e la riga unamed

results_paths = ['coarse_gs_results_giulia1.csv', 'coarse_gs_results_irene1.csv', 'coarse_gs_results_lorenzo1.csv',
                 'coarse_gs_results_giulia800_1.csv', 'coarse_gs_results_giulia800_2.csv', 'coarse_gs_results_giulia800_3.csv',
                 'coarse_gs_results_giulia800_4.csv', 'coarse_gs_results_irene2.csv']
all_results_path = 'coarse_gs_results_risultati_parziali.csv'
network_metric = 'mee'
val_metric = 'mse'
K = 3

# concatenate results into a single dataframe
scores_df = pd.DataFrame(columns=[])
for path in results_paths:
	partial_scores_df = pd.read_csv(path, sep=",")
	scores_df = pd.concat([scores_df, partial_scores_df], ignore_index=True)

# rank results
scores_df['val_%s_mean_rank'%network_metric] = rankdata(scores_df['val_%s_mean'%network_metric], method='dense')
scores_df['val_%s_mean_rank'%val_metric] = rankdata(scores_df['val_%s_mean'%val_metric], method='dense')
scores_df['tr_%s_mean_rank'%network_metric] = rankdata(scores_df['tr_%s_mean'%network_metric], method='dense')
scores_df['tr_mse_mean_rank'] = rankdata(scores_df['tr_mse_mean'], method='dense')

# sort results by 'val_score_mean_rank'
scores_df = scores_df.sort_values(by=['val_%s_mean_rank'%network_metric], ignore_index=True)
scores_df.drop(scores_df.columns[0], axis=1, inplace=True)

# change column order TODO: sistemare inserendo tutte le colonne
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
	columns_order.append('split%d_val_mse'%i)
	columns_order.append('split%d_val_mee'%i)
	columns_order.append('split%d_tr_mse'%i)
	columns_order.append('split%d_tr_mee'%i)
	columns_order.append('split%d_best_epoch'%i)
columns_order.append('params')
scores_df = scores_df[columns_order]

# write all results into a single csv
scores_df.to_csv(all_results_path)

scores_df.drop(scores_df.columns[0], axis=1, inplace=True) # drop first column
print(scores_df.iloc[0]) # first row
scores_dict = scores_df.to_dict(orient='records')
print(scores_dict[0]) # print row 0 as dict