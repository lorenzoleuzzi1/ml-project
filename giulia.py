from ensembler import Ensemble
from validation import read_csv_results
from cup import load_dev_set_cup
import pandas as pd
# from utils import save_obj

# cross validation ensemble
params = read_csv_results("csv_results/fine_gs2_results.csv")['params'][:10]

print(params)
X_dev, y_dev = load_dev_set_cup()

ens = Ensemble(params, 5)
cv_results = ens.validate(X_dev, y_dev, k=5)

results_path = 'cv_ens_results.csv'
df = pd.DataFrame([cv_results])
df.to_csv(results_path)



