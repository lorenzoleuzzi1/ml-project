from cup import load_dev_set_cup, load_internal_test_cup, read_tr_cup, load_blind_test_cup
from ensembler import Ensemble
from utils import read_csv_results, save_obj, mse, mee
import json
import pandas as pd
import time

# blind final model
params = read_csv_results("csv_results/fine_gs2_results.csv")['params'][:10]

X_tr, y_tr = read_tr_cup()
X_blind = load_blind_test_cup()

ens = Ensemble(params, 5)
start = time.time()
ens.fit(X_tr, y_tr)
end = time.time()
ens.plot()
preds = ens.predict(X_blind)
print(f"training time {end-time}")
print(preds)

df = pd.DataFrame(preds)
f = open('TheEnsemblers_ML-CUP22-TS.csv', 'a')
f.write('# Giulia Ghisolfi	Lorenzo Leuzzi	Irene Testa\n')
f.write('# TheEnsemblers\n')
f.write('# ML-CUP22\n')
f.write('# 23/01/2023\n')
df.to_csv(f, header=False, line_terminator='\r')
f.close()

save_obj(ens)
save_obj(preds)