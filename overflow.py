from network import Network

# SULLA CUP VA IN OVERFLOW!

net = Network(
    activation_out='identity',
    hidden_layer_sizes=[50, 50],
    classification=False,
    activation_hidden='tanh',
    loss='mse',
    evaluation_metric='mee',
    lambd=0.0001,
    alpha=0.9,
    epochs=500,
    batch_size=1.0, 
    learning_rate = "linear_decay",
    learning_rate_init=0.1, # TROPPO GRANDE! con 0.01 già meglio
    nesterov=True,
    stopping_patience=20,
    metric_decrease_tol=0.0001,
    validation_frequency=3,
    early_stopping=True
    )