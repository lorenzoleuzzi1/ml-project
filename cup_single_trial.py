from cup_parsing import load_dev_set_cup, load_internal_test_cup
from network import Network
from validation import k_fold_cross_validation
from sklearn.model_selection import ParameterGrid
import time
import cProfile
from utils import regression2_plots, mse, mee
import matplotlib.pyplot as plt

X_train, y_train = load_dev_set_cup()

net = Network(
    activation_out='identity',
    hidden_layer_sizes=[60,60],
    classification=False,
    activation_hidden='logistic',
    loss='mse',
    evaluation_metric='mee',
    epochs=800,
    lambd=0.0001,
    batch_size=1, 
    learning_rate = "fixed",
    learning_rate_init=0.0001,
    nesterov=False,
    stopping_patience=30,
    early_stopping=False,
    alpha=0.8
    )

net.fit(X_train, y_train)
X_test, y_test = load_internal_test_cup()
pred = net._predict_outputs(X_test)
print(mse(y_test, pred))
print(mee(y_test, pred))
regression2_plots(y_test, pred)
plt.plot(net.train_losses_reg)
plt.plot(net.train_scores)
plt.show()
#plt.plot(net.val_scores)

# print(X_train.shape[0])
#cProfile.run("net.fit(X_train, y_train)", sort="cumtime")
# start = time.time()
# net.fit(X_train, y_train)
# end = time.time()
# print(end-start)