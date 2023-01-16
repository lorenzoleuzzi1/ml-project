import numpy as np
import pandas as pd
import json
from network import Network
from validation import read_grid_search_results, k_fold_cross_validation
from cup_parsing import load_dev_set_cup

X_dev, y_dev = load_dev_set_cup()
df = read_grid_search_results("fine_gs2_results.csv")
best_n = 10
results_path = '/kaggle/working/k_fold_bests.csv'

df_scores = pd.DataFrame(columns=[])
for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    print(config)
    net = Network(**config)
    results = k_fold_cross_validation(net, X_dev, y_dev, k=5, evaluation_metric='mse')
    results['params'] = json.dumps(config)
    df_scores = pd.concat([df_scores, pd.DataFrame([results])], ignore_index=True)

df_scores.to_csv(results_path)