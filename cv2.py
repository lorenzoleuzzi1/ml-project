from validation import read_grid_search_results, k_fold_cross_validation
from network import Network
from cup_parsing import load_dev_set_cup
import pandas as pd
import json

# TODO: ripetizione per valutare influenza inizializzazione pesi
results_path = 'fine_gs2_repetition.csv'

df = read_grid_search_results("fine_gs2_results.csv")
X_train, y_train = load_dev_set_cup()

best_n = 5
preds = []

df_scores = pd.DataFrame(columns=[])
for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    net = Network(**config)
    cv_results = k_fold_cross_validation(net, X_train, y_train, k=5, evaluation_metric='mse')
    cv_results['params'] = json.dumps(config)
    df_scores = pd.concat([df_scores, pd.DataFrame([cv_results])], ignore_index=True)

df_scores.to_csv(results_path)