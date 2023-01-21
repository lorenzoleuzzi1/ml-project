# Machine Learning (ML) project
Final project for the Machine Learning course at University of Pisa, a.y. 2022/23. The project consists in implementing Neural Networks.

### Installation and Usage ###
`pip install -r requirements.txt`

The application allows to run different script using a basic command-line interface. To do so you need to type the following command `python main.py <script_name>` where `script_name` must be one of the following:
- `monks-1`: to fit and test a network with the monks-1 dataset;
- `monks-2`: to fit and test a network with the monks-2 dataset;
- `monks-3`: to fit and test a network with the monks-3 dataset;
- `monks-3reg`: to fit and test a network with the monks-3 dataset using L2 regularization;
- `cup-best_single`: to fit with the CUP development set and test with the CUP internal test set the network with the best configuration found after the fine grid search;
- `cup-custom`: to fit with the CUP development set and test with the internal test the network with a custom configuration properly defined in the cup_custom_config.json file;
- `cup-best_models_assessment`: to fit with the CUP development set and test with the CUP internal test set the networks with the best 10 models resulted from the fine gread search;
- `cup-ensemble_assessment`: to fit with the CUP development set and test with the CUP internal test set the ensemble;
- `cup-ensemble_cv`: to perform a k-fold cross validation with the ensemble;
- `cup-ensemble_final_fit`: to fit with the whole CUP training set the ensemble and save the object in a pickle file (for reproducibility);
- `cup-ensemble_blind_pred`: to load the final ensembled model and make the predictions on the cup blind test set;
- `cup-coarse_gs`: to perform the coarse grid search with the cup dataset;
- `cup-fine_gs`: to perform the fine grid search with the cup dataset.

### Directory Structure
```
ml-project
├── README.md
├── TheEnsemblers_ML-CUP22-TS.csv
├── csvs
│   ├── coarse_gs.csv
│   ├── coarse_gs_params_rank.csv
│   ├── cv_ens_results.csv
│   ├── fine_gs.csv
│   └── fine_gs_params_rank.csv
├── cup.py
├── datasets
│   ├── CUP_BLIND_TS.pkl
│   ├── CUP_DEV.pkl
│   ├── CUP_TS.pkl
│   ├── ML-CUP-short-info2022.txt
│   ├── ML-CUP22-TR.csv
│   ├── ML-CUP22-TS.csv
│   ├── monks-1.test
│   ├── monks-1.train
│   ├── monks-2.test
│   ├── monks-2.train
│   ├── monks-3.test
│   └── monks-3.train
├── ensemble.py
├── jsons
│   ├── best_models_assessment_scores.json
│   ├── ensemble_assessment_scores.json
│   ├── ensemble_cv_results.json
│   ├── grid_searches.json
│   └── monks_params.json
├── layer.py
├── main.py
├── monks.py
├── neural_network.py
├── pkls
│   ├── ens.pkl
│   └── preds.pkl
├── plots
│   ├── cup
│   │   ├── cup_loss.pdf
│   │   └── cup_score.pdf
│   ├── monks
│   │   ├── monks-1_loss.pdf
│   │   ├── monks-1_score.pdf
│   │   ├── monks-2_loss.pdf
│   │   ├── monks-2_score.pdf
│   │   ├── monks-3_loss.pdf
│   │   ├── monks-3_score.pdf
│   │   ├── monks-3reg_loss.pdf
│   │   └── monks-3reg_score.pdf
│   └── plots_report
│       ├── mean_all_models_losses.pdf
│       ├── mean_all_models_scores.pdf
│       ├── mean_final_losses.pdf
│       ├── mean_final_scores.pdf
│       ├── model_0_losses.pdf
│       ├── model_0_scores.pdf
│       ├── model_1_losses.pdf
│       ├── model_1_scores.pdf
│       ├── model_2_losses.pdf
│       ├── model_2_scores.pdf
│       ├── model_3_losses.pdf
│       ├── model_3_scores.pdf
│       ├── model_4_losses.pdf
│       ├── model_4_scores.pdf
│       ├── model_5_losses.pdf
│       ├── model_5_scores.pdf
│       ├── model_6_losses.pdf
│       ├── model_6_scores.pdf
│       ├── model_7_losses.pdf
│       ├── model_7_scores.pdf
│       ├── model_8_losses.pdf
│       ├── model_8_scores.pdf
│       ├── model_9_losses.pdf
│       ├── model_9_scores.pdf
│       ├── monks-1_accuracy.pdf
│       ├── monks-1_loss.pdf
│       ├── monks-2_accuracy.pdf
│       ├── monks-2_loss.pdf
│       ├── monks-3_accuracy.pdf
│       ├── monks-3_loss.pdf
│       ├── monks-3_reg_accuracy.pdf
│       └── monks-3_reg_loss.pdf
├── requirements.txt
├── utils.py
└── validation.py
```
### Code implementation
The three core classes that costitutes our code implementation are: `Layer`, `NeuralNetwork` and for the final model `Ensemble`. And they're defined as follows:

`Layer`
```
    A class representing a feed-forward fully connected layer of a neural network.
    It implements the methods needed to propagate input values to the units of the next layer and to backpropagate the errors coming from them.

    Attributes:
        - input (ndarray): inputs values to the units of the layer.   
        - net (ndarray): net input values to the units of the layer.  
        - output (ndarray): outputs values of the units of the layer. 
        - fan_in (int): number of inputs to each unit of the layer.
        - fan_out (int): Number of outputs of each unit of the layer.
        - activation (str): activation function name.
        - activation_fun (function): activation function.
        - activation_prime (function): derivative of the activation function.
        - weights (ndarray): current weights values associated to the incoming links of the units.
        - bias (ndarray): current bias value associated to each unit.
        - init_weights (ndarray): initial weights values associated to the incoming links of the units.
        - init_bias (ndarray): initial bias value associated to each unit.
        - deltas_weights (ndarray): gradient of the error w.r.t the weights.
        - deltas_bias (ndarray): gradient of the error w.r.t the biases.
        - velocity_w (ndarray): weights velocity term to apply momentum.
        - velocity_b (ndarray): bias velocity term to apply momentum.
```
That exposes the following methods:
```
    def forward_propagation(self, input_data):
        Performs the forward pass.
        
        Parameters:
            - input_data (ndarray): layer's inputs.
        
        Returns:
            - output (ndarray): layer's outputs.
```
```
    def backward_propagation(self, delta_j):
        Performs the backward pass.

        Parameters:
            - delta_j (ndarray): incoming error (from the units in the next layer j).

        Returns:
            - delta_i ((ndarray): outcoming error (from the units in the current layer i).
```
```
    def update(self, learning_rate, batch_size, alpha, lambd, nesterov):
        Update weights and biases with the accumulated deltas.

        Parameters:
            - learning_rate (float): learning rate value.
            - batch_size (int): number of patterns in the batch.
            - alpha (float): momentum coefficient.
            - lambd (float): L2 regularization coefficient.
            - nesterov (bool): wheter to apply Nesterov momentum.
```

`NeuralNetwork`
```
    A neural network class for classification or regression tasks.

    Attributes:
        - activation_out (str): the activation function to use for the output layer. Choices: 'identity', 'relu', 'leaky_relu', 'logistic', 'tanh', 'softplus', 'softmax'.
        - classification (bool): a boolean indicating whether the task is classification or regression.
        - activation_hidden (str): the activation function to use for the hidden layers. Choices: 'identity', 'relu', 'leaky_relu', 'logistic', 'tanh', 'softplus', 'softmax'.
        - hidden_layer_sizes (list): a list of integers representing the number of neurons in each hidden layer.
        - loss (str): the loss function to use for the network. Choices: 'mse', 'mee', 'logloss'.
        - evaluation_metric (str): the evaluation metric to use for the network. Choices: 'mse', 'mee', 'logloss', 'accuracy'.
        - epochs (int): the number of training iterations to perform.
        - learning_rate (str): the type of learning rate to use. Choices: 'fixed', 'linear_decay'.  
        - learning_rate_init (float): the initial learning rate value.
        - tau (int): the number of iterations over the training data before the learning rate is decreased. Only used if learning_rate = 'linear_decay'.
        - batch_size (int or float): the number of samples to use in each training update.
        - ambd (float): the regularization term to use as weight decay.
        - alpha (float): the momentum term to use in the optimizer.
        - nesterov (bool): a boolean indicating whether to use Nesterov momentum.
        - early_stopping (bool): a boolean indicating whether to use early stopping during training.
        - stopping_criteria_on_loss (bool): a boolean indicating whether to stop training if the loss stops decreasing.
        - stopping_patience (int): the number of iterations to wait for improvement before stopping training.
        - validation_size (int or float): the proportion of the data to use for validation during training.
        - tol (float): the tolerance for the optimizer.
        - metric_decrease_tol (float): the tolerance for the evaluation metric during early stopping.
        - verbose (bool): a boolean indicating whether to print progress during training.
        - random_state (int): the seed to use for the random number generator.
        - reinit_weights (bool): a boolean indicating whether to reinitialize the weights at the beginning of training.
        - weights_dist (str): the distribution to use for initializing the weights. Choices: 'normal', 'uniform'.
        - weights_bound (float): the range for the distribution used to initialize the weights.
        - train_losses (list of float): a list of training losses for each epoch.
        - val_losses (list of float): a list of validation losses for each epoch.
        - train_scores (list of float): a list of training scores for each epoch.
        - val_scores (list of float): a list of validation scores for each epoch.
```
That exposes the following methods:
```
    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        Train the neural network on the given training data.

        Parameters:
            - X_train (np.array): the training data.
            - Y_train (np.array): the target data for the training set.
            - X_val (np.array): the validation data. Optional, if not provided, no validation score will be calculated.
            - Y_val (np.array): the target data for the validation set.
```
```
    def predict(self, X):
        Make predictions for the given data.

        Parameters:
            - X (np.array): the data to make predictions for.

        Returns:
            - predictions (np.array): the predictions made by the network.
```
```
    def score(self, X_test, Y_test, evaluation_metrics):
            Evaluate the performance of the neural network on the given test data.

            Parameters:
                - X_test (np.array): the test data to evaluate the network on.
                - Y_test (np.array): the true target data for the test set.
                - evaluation_metrics (str): a list of the evaluation metrics to use.

            Returns:
                - scores (float): the score of the network on the test data according to the chosen evaluation metrics.
```
`Ensemble`
```
    Ensemble of Neural Networks

    Attributes:
        - models_params (list of dict): list of dictionaries, where each dictionary contains the parameters for instantiating a neural network.    
        - n_trials (int): number of trials to run to obtain the ensemble's prediction.
        - models  (list of NeuralNetwork): list of the neural network models in the ensemble.
        - train_losses_trials_mean (list of float): a list of the mean of training losses for each epoch over the trials for the same model.
        - val_losses_trials_mean (list of float): a list of the mean of training scores for each epoch over the trials for the same model.
        - train_scores_trials_mean (list of float): a list of the mean of validation losses for each epoch over the trials for the same model.
        - val_scores_trials_mean (list of float): a list of the mean of validation scores for each epoch over the trials for the same model.
        - train_losses__mean (list of float): a list of the mean of training losses for each epoch over the models.
        - val_losses_mean (list of float): a list of the mean of training losses for each epoch over the models.
        - train_scores_mean (list of float): a list of the mean of training losses for each epoch over the models.
        - val_scores_mean (list of float): a list of the mean of training losses for each epoch over the models.
        - final_train_loss (float): the mean of the best training losses.
        - final_val_loss (float): the mean of the best training scores.
        - final_train_score (float): the mean of the best validation losses.
        - final_val_score (float): the mean of the best validation scores.
```
That exposes the following methods:
```
    def fit(self, X_train, y_train, X_test = None, y_test = None):
        Train the ensemble of neural networks for each param's configuration in the model_params list on 
        the provided data
        
        Parameters:
            - X_train (np.array): the training data.
            - Y_train (np.array): the target data for the training set.
            - X_test (np.array): the test data. Optional, if not provided, no validation score will be calculated.
            - Y_test (np.array): the target data for the test set. Optional, if not provided, no validation score will be calculated.
```
```
    def predict(self, X):
        Make the ensemble's predictions for the given data.
        Parameters:
            - X (np.array): the data to make predictions for.

        Returns:
            - preds (np.array): ensemble's prediction for each sample in X
```

```
def validate(self, X_train, y_train, k):
        Model assessment through a k-fold cross validation for each model of the ensemble.
        
        Parameters: 
            - X_train (np.array): the training data.
            - y_train (np.array): the target data for the training set.
            - k (int): the number of folds to use for cross validation.

        Returns:
            - results (dict): a dict of the result obtained in the k-fold cross validation process.
```
