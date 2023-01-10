import pandas as pd
from scipy.stats import rankdata

results_paths = ['coarse_gs_results_giulia.csv', 'coarse_gs_results_irene.csv', 'coarse_gs_results_lorenzo.csv']
all_results_path = 'coarse_gs_results.csv'
K = 3

# concatenate results into a single dataframe
scores_df = pd.DataFrame(columns=[])
for path in results_paths:
	partial_scores_df = pd.read_csv(path, sep=",")
	scores_df = pd.concat([scores_df, partial_scores_df], ignore_index=True)

# rank results
scores_df['val_score_mean_rank'] = rankdata(scores_df['val_score_mean'], method='dense')
scores_df['tr_score_mean_rank'] = rankdata(scores_df['tr_score_mean'], method='dense')
scores_df['tr_loss_mean_rank'] = rankdata(scores_df['tr_loss_mean'], method='dense')

# sort results by 'val_score_mean_rank'
scores_df.sort_values(by=['val_score_mean_rank'])

# change column order
columns_order = [
	'val_score_mean',
	'val_score_mean_rank',
	'tr_loss_mean',
	'tr_loss_mean_rank',
	'tr_score_mean',
	'tr_score_mean_rank'
]
for i in range(K):
	columns_order.append('split%d_val_score'%i)
	columns_order.append('split%d_tr_loss'%i)
	columns_order.append('split%d_tr_score'%i)
	columns_order.append('split%d_best_epoch'%i)
columns_order.append('params')
scores_df = scores_df[columns_order]

# write all results into a single csv
scores_df.to_csv(all_results_path)

scores_df.drop(scores_df.columns[0], axis=1, inplace=True) # drop first column
print(scores_df.iloc[0]) # first row
scores_dict = scores_df.to_dict(orient='records')
print(scores_dict[0]) # print row 0 as dict