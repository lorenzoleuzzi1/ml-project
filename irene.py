from cup import load_dev_set_cup, load_internal_test_cup
from ensembler import Ensemble
from utils import read_csv_results, save_obj, mse, mee
import json
import time 

# assessment ensemble
params = read_csv_results("csv_results/fine_gs2_results.csv")['params'][:10]

X_dev, y_dev = load_dev_set_cup()
X_test, y_test = load_internal_test_cup()

ens = Ensemble(params, 5)
start = time.time()
ens.fit(X_dev, y_dev, X_test, y_test)
end = time.time()
ens.plot()
preds = ens.predict(X_test)
print("training time %f"%(end-start))
mse_score = mse(y_test, preds)
mee_score = mee(y_test, preds)

print(preds)
print(mse_score)
print(mee_score)

save_obj(ens, "ensemble.pkl")
save_obj(preds, "internal_preds.pkl")

d = {
    "mse" : mse_score,
    "mee" : mee_score
}

with open("assessment_scores.json", 'w') as f:
	json.dump(d, fp = f, indent = 4)