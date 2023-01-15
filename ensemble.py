from validation import read_grid_search_results
from network import Network
from cup_parsing import load_dev_set_cup
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


df = read_grid_search_results("fine_gs2_results.csv")
X_train, y_train = load_dev_set_cup()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
#print(df)
#l = [31, 28, 35]
best_n = 5
preds = []
for i in range(best_n):
    print(f"{i}/{best_n}")
    config = df['params'][i]
    print(config)
    
    for i in range(5):
        net = Network(**config)
        net.fit(X_train, y_train)
        #net.early_stopping = True
        #net.stopping_patience = 500
        #net.validation_size = 0.2
        preds.append(net.predict_outputs(X_val))    
    
    # plt.figure(figsize=(30,20))
    # #plt.plot(net.train_losses, )
    # plt.semilogy(net.train_scores, color='pink')
    # plt.semilogy(net.val_scores, color='green')
    # plt.savefig(f"fine2_{i}_model.pdf") 
file = open('ensemble.pkl', 'wb')
pickle.dump(preds, file)
file.close()
# y1 = preds[:][0]
# y2 = preds[:][1]
