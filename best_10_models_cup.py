from network import Network
from utils import read_csv_results
from cup import load_dev_set_cup, load_internal_test_cup
import time

configs = read_csv_results("csv_results/fine_gs2_results.csv")['params'][:10]
X_train, y_train = load_dev_set_cup()
X_test, y_test = load_internal_test_set_cup()
scores = []
for config in configs:
    #print(f"Running cup with the following configuration:\n{config}")
    X_train, y_train = load_dev_set_cup()
    net = Network(**config)

    start = time.time()
    net.fit(X_train, y_train)
    end = time.end()
    print(end-time)
    score = net.score(X_test, y_test, ['mse', 'mee'])
    print(score)
    scores.append(score)

print(scores)



