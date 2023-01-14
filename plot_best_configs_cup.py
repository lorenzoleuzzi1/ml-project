from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup
import matplotlib.pyplot as plt


df = read_grid_search_results("fine_gs_edited.csv")
X_train, y_train = load_dev_set_cup()
print(df)
best_n = 10
for i in range(10):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    net = Network(**config)
    net.early_stopping = True
    net.stopping_patience = 500
    net.validation_size = 0.2
    net.fit(X_train, y_train)
    plt.figure(figsize=(30,10))
    #plt.plot(net.train_losses, )
    plt.semilogy(net.train_scores, color='pink')
    plt.semilogy(net.val_scores, color='green')
    plt.savefig(f"fine_{i}_model.pdf") 
