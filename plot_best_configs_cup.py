from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup
import matplotlib.pyplot as plt


df = read_grid_search_results("coarse_gs.csv")
X_train, y_train = load_dev_set_cup()
print(df)

for i in range(10):
    config = df['params'][0]
    net = Network(**config)
    net.fit(X_train, y_train)
    plt.plot(net.train_losses)
    plt.plot(net.train_scores)  
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(20)
    plt.savefig(f"{i}_model.pdf")
