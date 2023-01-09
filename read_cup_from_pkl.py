from cup_parsing import load_blind_test_cup, load_dev_set_cup, load_internal_test_cup
from network import Network

X_blind_test = load_blind_test_cup()
# print("X blind")
# print(X_blind_test)
X_dev, y_dev = load_dev_set_cup()
# print("X dev")
# print(X_dev)
# print("y dev")
# print(y_dev)
X_test, y_test = load_internal_test_cup()
# print("X test")
# print(X_test)
# print("y test")
# print(y_test)

di = {'activation_hidden': 'tanh', 'activation_out': 'identity', 'alpha': 0.9, 'batch_size': 256, 'classification': False, 'early_stopping': False, 'epochs': 500, 'evaluation_metric': 'mee', 'hidden_layer_sizes': [10, 10], 'lambd': 0.0001, 'learning_rate': 'linear_decay', 'learning_rate_init': 0.0005, 'loss': 'mse', 'metric_decrease_tol': 0.001, 'nesterov': False, 'random_state': None, 'reinit_weights': True, 'stopping_patience': 5, 'tau': 200, 'tol': 0.0001, 'validation_size': 0.1, 'verbose': True, 'weights_dist': None}
# net = Network(
#     activation_out='identity',
#     classification=False,
#     activation_hidden='leaky_relu',
#     loss='mse',
#     evaluation_metric='mee',
#     epochs=500,
#     batch_size=32, 
#     learning_rate = "fixed",
#     learning_rate_init=0.001,
#     nesterov=False,
#     stopping_patience=10,
#     early_stopping=True
#     )

net = Network(**di)
net = Network(
    activation_out='identity',
    hidden_layer_sizes=[50, 50],
    classification=False,
    activation_hidden='tanh',
    loss='mse',
    evaluation_metric='mee',
    lambd=0.0001,
    alpha=0.9,
    epochs=100,
    tau = 10,
    batch_size=1.0, 
    learning_rate = "linear_decay",
    learning_rate_init=10, # TROPPO GRANDE! con 0.01 già meglio
    nesterov=True,
    stopping_patience=100,
    metric_decrease_tol=0.0001,
    early_stopping=True
    )
net.fit(X_dev, y_dev)
pred = net.predict(X_test)
print(net.evaluate(Y_true=y_test, Y_pred=pred))