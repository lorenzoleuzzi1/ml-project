from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup
import matplotlib.pyplot as plt


df = read_grid_search_results("coarse_gs.csv")
X_train, y_train = load_dev_set_cup()
print(df)

for i in range(6,10):
    config = df['params'][0]
    net = Network(**config)
    net.fit(X_train, y_train)
    
    plt.figure()
    plt.figure(figsize=(20,20))
    plt.plot(net.train_losses)
    plt.plot(net.train_scores)
    plt.savefig(f"{i}_model.pdf")
    print("done!")
