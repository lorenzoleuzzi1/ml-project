from ensembler import Ensemble
from utils import read_csv_results
from validation import grid_search_cv
from cup import load_dev_set_cup, load_internal_test_cup
from monks import read_monks
import numpy as np
import pandas as pd
import pickle
import json

params = read_csv_results("csv_results/fine_gs2_results.csv")['params'][:2]

ens = Ensemble(params, 2)
X_train, y_train = load_dev_set_cup()
#X_test, y_test = load_internal_test_cup()

ens.fit(X_train, y_train)
pred = ens.predict(X_train)
print(pred)

#cv_results = ens.validate(X_train, y_train, k=3)

"""
results_path = 'cv_ens_results.csv'
df = pd.DataFrame([cv_results])
df.to_csv(results_path)"""


#ens.fit(X_train, y_train, X_test, y_test)

"""preds = ens.predict(X_train[:10])
print(preds)
ens.plot()"""
# file = open("test_ens.pkl", 'wb')
# pickle.dump(ens, file)
# file.close()

# file = open('test_ens.pkl', 'rb')
# data = pickle.load(file)
# file.close()
# preds = data.predict(X_train[:10])
# print(preds)

# preds_monks = [[1,2,3],[2,2,2]]
# preds_cup = [[[1,2],[1,2]],[[2,2],[2,2]]]

# mean_pred_monks = np.mean(np.array(preds_monks), axis=0)
# mean_pred_cup = np.mean(np.array(preds_cup), axis=0)

# print(mean_pred_monks)
# print(mean_pred_cup)

