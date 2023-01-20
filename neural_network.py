import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
from math import floor, ceil
from utils import *
from layer import Layer
from sklearn import preprocessing


class NeuralNetwork:
    """
    A neural network class for classification or regression tasks.

    Parameters:
        - activation_out (str): the activation function to use for the output layer. 
            Choices: 'identity', 'relu', 'leaky_relu', 'logistic', 'tanh', 'softplus', 'softmax'
        
        - classification (bool): a boolean indicating whether the task is classification or regression.
        
        - activation_hidden (str): the activation function to use for the hidden layers. 
            Choices: 'identity', 'relu', 'leaky_relu', 'logistic', 'tanh', 'softplus', 'softmax'
        
        - hidden_layer_sizes (list): a list of integers representing the number of neurons in each hidden layer.
        
        - loss (str): the loss function to use for the network. 
            Choices: 'mse', 'mee', 'logloss'
        
        - evaluation_metric (str): the evaluation metric to use for the network.
            Choices: 'mse', 'mee', 'logloss', 'accuracy'
        
        - epochs (int): the number of training iterations to perform.
        
        - learning_rate (str): the type of learning rate to use. 
            Choices: 'fixed', 'linear_decay'
        
        - learning_rate_init (float): the initial learning rate value.
        
        - tau (int): the number of iterations over the training data before the learning rate is decreased.
            Only used if learning_rate = 'linear_decay'
        
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
        
        - weights_dist (str): the distribution to use for initializing the weights.
            Choices: 'normal', 'uniform'
        
        - weights_bound (float): the range for the distribution used to initialize the weights.
    """
    def __init__(
        self, 
        activation_out : str,
        classification : bool,
        activation_hidden : str ='tanh',
        hidden_layer_sizes = [3],
        loss : str ='mse',
        evaluation_metric : str = 'mse',
        epochs : int = 200,
        learning_rate : str = 'fixed',
        learning_rate_init : float = 0.0001,
        tau : int = 200,
        batch_size : int or float = 1,
        lambd : float = 0.0001,
        alpha : float = 0.9,
        nesterov : bool = False,
        early_stopping : bool = False,
        stopping_criteria_on_loss : bool = True,
        stopping_patience : int = 20,
        validation_size : int or float = 0.1,
        tol : float = 0.00001,
        metric_decrease_tol : float = 0.00001,   
        verbose : bool = True,
        random_state = None,
        reinit_weights : bool = True,
        weights_dist : str = None,
        weights_bound : float = None  
        ):
       
        self._check_params(locals())
        self.layers = []
        self.first_fit = True
        self.activation_out = activation_out
        self.activation_hidden = activation_hidden
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss = loss
        self.loss_fun = LOSSES[loss]
        self.loss_prime = LOSSES_DERIVATIVES[loss]
        self.evaluation_metric = evaluation_metric
        self.evaluation_metric_fun = EVALUATION_METRICS[evaluation_metric]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.learning_rate_curr = learning_rate_init
        self.learning_rate_fin = learning_rate_init * 0.1
        self.tau = tau
        self.batch_size = batch_size
        self.lambd = lambd
        self.alpha = alpha
        self.nesterov = nesterov
        self.stopping_patience = stopping_patience
        self.early_stopping = early_stopping
        self.stopping_criteria_on_loss = stopping_criteria_on_loss
        self.validation_size = validation_size
        self.tol = tol
        self.metric_decrease_tol = metric_decrease_tol
        self.classification = classification
        self.verbose = verbose
        self.random_state = random_state
        self.reinit_weights = reinit_weights
        self.weights_dist = weights_dist # None, 'normal' or 'uniform'
        self.weights_bound = weights_bound # if 'normal' is the std, if 'uniform' in [-weights_bound, weights_bound]   
        if self.activation_out == 'tanh': self.neg_label = -1.0
        else: self.neg_label = 0.0
        self.pos_label = 1.0

    def _check_params(self, params):
        if (params['activation_out'] not in ACTIVATIONS):
            raise ValueError("Unrecognized activation_out. "
                "Supported activation functions are %s." % list(ACTIVATIONS))
        if (params['activation_hidden'] not in ACTIVATIONS):
            raise ValueError("Unrecognize activation_hidden. "
                "Supported activation functions are %s." % list(ACTIVATIONS))
        if not isinstance(params['hidden_layer_sizes'], list):
            raise ValueError("hidden_layer_sizes must be a list of integers.")
        if any(size <= 0 for size in params['hidden_layer_sizes']):
            raise ValueError("hidden_layer_sizes must be > 0.")
        if params['loss'] not in LOSSES:
            raise ValueError("Unrecognized loss. "
                "Supported losses functions are %s." % list(LOSSES))
        if not isinstance(params['classification'], bool):
            raise ValueError("classification must be a boolean.")
        if params['classification'] == False and params['loss'] == 'logloss':
            raise ValueError("Cannot use logloss for a regression task.")
        if params['loss'] == 'logloss' and params['activation_out'] != 'softmax':
            raise ValueError("logloss must be used with activation_out='softmax'.")
        if params['classification'] == True and params['activation_out'] in ['identity', 'relu', 'leaky_relu', 'sofplus']:
            raise ValueError("Cannot use activation_out='%s' for a classification task." % params['activation_out'])
        if params['classification'] == False and params['activation_out'] in ['logistic', 'tanh', 'softmax']:
            raise ValueError("Cannot use activation_out='%s' for a regression task." % params['activation_out'])
        if params['epochs'] <= 0:
            raise ValueError("epochs must be > 0.")
        if params['evaluation_metric'] not in EVALUATION_METRICS:
            raise ValueError("Unrecognized evaluation metric. "
                "Supported evaluation metrics are %s."% list(EVALUATION_METRICS))
        if params['evaluation_metric'] == 'accuracy' and params['classification'] == False:
            raise ValueError("accuracy metric can be used only for classification tasks.")
        if params['learning_rate'] not in ["fixed", "linear_decay"]:
            raise ValueError("Unrecognized learning_rate_schedule. "
            "Supported learning rate schedules are %s." % ["fixed", "linear_decay"])
        if params['learning_rate_init'] <= 0.0:
            raise ValueError("learning_rate_init must be > 0.")
        if params['learning_rate'] == "linear_decay":
            if params['tau'] <= 0 or params['tau'] > params['epochs']:
                raise ValueError("tau must be > 0 and <= epochs.")
        if (not isinstance(params['batch_size'], int)) and \
            (not isinstance(params['batch_size'], float)):
            raise ValueError("batch_size must be a float or an integer.")
        if params['batch_size'] <= 0:
            raise ValueError("batch_size must be > 0.")
        if params['lambd'] < 0.0:
            raise ValueError("lambd must be >= 0.")
        if params['alpha'] > 1 or params['alpha'] < 0:
            raise ValueError("alpha must be >= 0 and <= 1.")
        if not isinstance(params['verbose'], bool):
            raise ValueError("verbose must be a boolean.")
        if not isinstance(params['nesterov'], bool):
            raise ValueError("nesterov must be a boolean.")
        if params['stopping_patience'] > params['epochs'] or params['stopping_patience']  <= 0:
            raise ValueError("patience must be between 1 and max epochs %s." % (params['epochs']))
        if not isinstance(params['early_stopping'] , bool):
            raise ValueError("ealry_stopping must be a boolean.")
        if params['validation_size'] <= 0:
            raise ValueError("validation_size must be > 0.")
        if params['tol'] < 0 or params['tol'] > 0.5:
            raise ValueError("tol must be > 0 and < 0.5")
        if params['random_state'] != None and not isinstance(params['random_state'], int):
            raise ValueError("random_state must be an integer.")
        if not isinstance(params['reinit_weights'], bool):
            raise ValueError("reinit_weights must be a boolean.")
        if params['weights_dist'] != None and not isinstance(params['weights_dist'], str):
            raise ValueError("weights_dist must be a string.")
        if params['weights_dist'] != None and not params['weights_dist'] in ['uniform', 'normal']:
            raise ValueError("Unrecognized weights_dist. "
            "Supported weights distributions are ['uniform', 'normal'].")
        if params['weights_bound'] != None and \
            not isinstance(params['weights_bound'], int) and \
            not isinstance(params['weights_bound'], float):
            raise ValueError("weights_bound must be an int or a float.")
        if params['metric_decrease_tol'] < 0:
            raise ValueError("metric_decrease_tol must be positive.")
        if (not isinstance(params['stopping_criteria_on_loss'], bool)):
            raise ValueError("stopping_criteria_on_loss must be a boolean.")

    def _encode_targets(self, Y_train):
        self.binarizer = preprocessing.LabelBinarizer(
            pos_label=self.pos_label,
            neg_label=self.neg_label
        )
        self.binarizer.fit(Y_train)
        if not self.first_fit and any(self.binarizer.classes_ != self.labels):
            self.first_fit = True
        self.labels = self.binarizer.classes_
        self.n_classes = len(self.binarizer.classes_)
        Y_train = self.binarizer.transform(Y_train).astype(np.float64)
        if self.n_classes == 2 and self.activation_out == 'softmax':
            Y_train = np.hstack((Y_train, 1 - Y_train))
        
        return Y_train

    # predict output for given input
    def _predict_outputs(self, X):
        Y = np.empty((X.shape[0], 1, self.n_outputs))
        # run network over all samples
        for i in range(X.shape[0]):
            # forward propagation
            output = X[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            Y[i] = output
        Y = Y.reshape(Y.shape[0], Y.shape[2])
        return Y
    
    def _discretize_outputs(self, Y):
        if self.n_classes == 2 and self.activation_out != 'softmax':
            Y_disc = np.where(
                Y > ACTIVATIONS_THRESHOLDS[self.activation_out],
                self.pos_label,
                self.neg_label)
        else:
            Y_disc = np.full_like(Y, self.neg_label)
            if Y.ndim == 1:
                max_idxs = np.argmax(Y)
                Y_disc[max_idxs] = self.pos_label
            else:
                max_idxs = np.argmax(Y, axis=1)
                Y_disc[np.indices(max_idxs.shape)[0], max_idxs] = self.pos_label
        return Y_disc

    def _outputs_to_labels(self, Y):
        Y_lbl = self._discretize_outputs(Y)
        if self.n_classes == 2 and self.activation_out == 'softmax':
            Y_lbl = Y_lbl[:, 0]
        Y_lbl = self.binarizer.inverse_transform(Y_lbl).astype(np.float64)
        Y_lbl = Y_lbl.reshape(Y_lbl.shape[0], 1)
        return Y_lbl

    def _evaluate(self, Y_true, Y_pred, metric):
        if metric == 'accuracy':
            Y = self._discretize_outputs(Y_pred)
        else:
            Y = Y_pred

        return EVALUATION_METRICS[metric](y_true=Y_true, y_pred=Y)

    # add layer to network
    def _add(self, layer):
        self.layers.append(layer)

    def _compose(self):
        if not self.first_fit and not \
            (self.layers[0].fan_in == self.n_features and \
            self.layers[-1].fan_out == self.n_outputs):
            self.first_fit = True
        
        if self.first_fit:
            self.layers = []
            # Add first hidden layer
            self._add(Layer(
                fan_in = self.n_features,
                fan_out = self.hidden_layer_sizes[0],
                activation = self.activation_hidden,
                weights_dist = self.weights_dist,
                weights_bound = self.weights_bound
                ))
            # Add further hidden layers
            for i in range(len(self.hidden_layer_sizes)-1):
                self._add(Layer(
                    fan_in = self.hidden_layer_sizes[i],
                    fan_out = self.hidden_layer_sizes[i+1],
                    activation = self.activation_hidden,
                    weights_dist = self.weights_dist,
                    weights_bound = self.weights_bound
                ))
            # Add output layer
            self._add(Layer(
                fan_in = self.hidden_layer_sizes[-1],
                fan_out = self.n_outputs,
                activation = self.activation_out,
                weights_dist = self.weights_dist,
                weights_bound = self.weights_bound
                ))
            self.first_fit = False
        elif self.reinit_weights:
            for layer in self.layers:
                layer.weights_init( self.weights_dist, self.weights_bound)

    def _update_learning_rate(self, epoch):
        if self.learning_rate == "fixed":
            self.learning_rate_curr = self.learning_rate_init
        
        if self.learning_rate == "linear_decay":
            a = epoch / self.tau
            lr = (1 - a) * self.learning_rate_init + a * self.learning_rate_fin
            
            if epoch == 0:
                self.learning_rate_curr = self.learning_rate_init
            elif epoch >= self.tau or lr < self.learning_rate_fin:
                self.learning_rate_curr = self.learning_rate_fin
            else:
                self.learning_rate_curr = lr

    def _update_no_improvement_count(self, epoch, train_losses, train_scores, val_scores):
        if epoch < 10:
            self.best_epoch = epoch
            self.best_loss = train_losses[-1] 
            self.best_metric = val_scores[-1] if self.early_stopping else train_scores[-1]
            self.best_weights, self.best_bias = self.get_current_weights()
            return
        
        if self.early_stopping:
            if self.evaluation_metric == 'accuracy':
                converged = val_scores[-1] >= 1-self.tol
                best_metric_delta = val_scores[-1] - self.best_metric
            else:
                converged = val_scores[-1] <= self.tol
                best_metric_delta = self.best_metric - val_scores[-1]
        else:
            converged = train_losses[-1] <= self.tol
            if self.stopping_criteria_on_loss:
                best_metric_delta = self.best_loss - train_losses[-1]
            else:
                if self.evaluation_metric == 'accuracy':
                    best_metric_delta = train_scores[-1] - self.best_metric
                else:
                    best_metric_delta = self.best_metric - train_scores[-1]

        if best_metric_delta > 0:
            self.best_epoch = epoch
            self.best_loss = train_losses[-1]
            self.best_metric = val_scores[-1] if self.early_stopping else train_scores[-1]
            self.best_weights, self.best_bias = self.get_current_weights()
        if converged:
            self.no_improvement_count = self.stopping_patience # if we've already converged (error near 0)
        elif best_metric_delta < self.metric_decrease_tol:
            self.no_improvement_count += 1 # if no significant improvement
        else:
            self.no_improvement_count = 0

    def _fit_preprocessing(self, X_train, Y_train, X_val, Y_val):
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2-dimensional array")
        if Y_train.ndim != 2:
            raise ValueError("Y_train must be a 2-dimensional array")
        if self.classification and Y_train.shape[1] > 1:
            raise ValueError("Multilabel classification is not supported.")
        if self.batch_size > X_train.shape[0]:
            raise ValueError("batch_size must not be larger than sample size.")
        if (X_val is None and Y_val is not None):
            raise ValueError("X_val is None.")
        if (X_val is not None and Y_val is None):
            raise ValueError("Y_val is None.")
        if X_val is not None:
            if X_val.ndim != 2:
                raise ValueError("X_val must be a 2-dimensional array")
            if Y_val.ndim != 2:
                raise ValueError("Y_val must be a 2-dimensional array")
            if Y_train.shape[1] != Y_val.shape[1]:
                raise ValueError("Y_train and Y_val do not have matching sizes.")
            if self.classification:
                train_labels = unique_labels(Y_train)
                val_labels = unique_labels(Y_val)
                if len(val_labels) > len(train_labels):
                    raise ValueError("validation labels are more than train labels.")
                for label in val_labels:
                    if label not in train_labels:
                        raise ValueError("validation labels are not included in train labels.") 
                self.labels = train_labels

        self.n_features = X_train.shape[1]
        if self.classification:
            Y_train = self._encode_targets(Y_train) 
            if Y_val is not None:
                Y_val = self.binarizer.transform(Y_val).astype(np.float64)
                if self.n_classes == 2 and self.activation_out == 'softmax':
                    Y_val = np.hstack((Y_val, 1 - Y_val))
            self.n_outputs = self.n_classes
            if self.n_classes == 2 and self.activation_out != 'softmax':
                self.n_outputs = 1
        else:
            self.n_outputs = Y_train.shape[1]

        return Y_train, Y_val

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        """
        Train the neural network on the given training data.

        Parameters:
            - X_train (np.array): the training data.

            - Y_train (np.array): the target data for the training set.

            - X_val (np.array): the validation data. Optional, if not provided, no validation score will be calculated.

            - Y_val (np.array): the target data for the validation set.
        """
        Y_train, Y_val = self._fit_preprocessing(X_train, Y_train, X_val, Y_val)
        self._compose()
        n_samples = X_train.shape[0]

        # early stopping validation split
        if self.early_stopping:
            if self.classification:
                stratify = Y_train
            else:
                stratify = None
            if Y_val is None:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train,
                    Y_train,
                    test_size=self.validation_size,
                    shuffle=True,
                    stratify=stratify,
                    random_state=self.random_state
                )
        
        # divide training set into batches
        if isinstance(self.batch_size, int):
            n_batches = ceil(n_samples / self.batch_size)
        else:
            n_batches = floor(1 / self.batch_size)

        self.train_losses_reg = []
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []

        self.no_improvement_count = 0

        # loop through the number of epochs
        for epoch in range(self.epochs):
            train_loss = 0
            train_loss_not_reg = 0
            train_score = 0
            X_train, Y_train = shuffle(X_train, Y_train, random_state=self.random_state)
            X_train_batched = np.array_split(X_train, n_batches)
            Y_train_batched = np.array_split(Y_train, n_batches)
            
            # loop through batches
            for X_batch, Y_batch in zip(X_train_batched, Y_train_batched):
                
                # loop through patterns in the batch
                for x, y in zip(X_batch, Y_batch):
                    batch_size = X_batch.shape[0]
                    output = x
                    
                    # forward propagation
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                      
                    # compute loss and evaluation metric 
                    train_loss += self.loss_fun(y_true=y, y_pred=output)
                    train_score += self._evaluate(Y_true=y, Y_pred=output, metric=self.evaluation_metric)
                    
                    # backward propagation
                    delta = self.loss_prime(y_true=y, y_pred=output)
                    for layer in reversed(self.layers):
                        delta = layer.backward_propagation(delta)

                # add l2 regularization term to the loss
                train_loss_not_reg = train_loss
                reg_term = 0
                for layer in self.layers:
                    weights = layer.weights.ravel()
                    reg_term += np.dot(weights, weights)
                reg_term = self.lambd * reg_term
                train_loss += reg_term

                # new learning rate
                self._update_learning_rate(epoch)
                # update weights
                for layer in self.layers:
                    layer.update(
                        learning_rate=self.learning_rate_curr,
                        batch_size=batch_size,
                        alpha=self.alpha,
                        lambd = self.lambd*(batch_size/n_samples),
                        nesterov=self.nesterov
                    )
            
            #-----validation-----
            if self.early_stopping or Y_val is not None:
                Y_val_output = self._predict_outputs(X_val)
                val_loss = self.loss_fun(y_true=Y_val, y_pred=Y_val_output)
                val_score = self._evaluate(Y_true=Y_val, Y_pred=Y_val_output, metric=self.evaluation_metric)
                self.val_losses.append(val_loss)
                self.val_scores.append(val_score)
            #--------------------

            # average on all samples 
            train_loss_not_reg /= n_samples
            train_loss /= n_samples
            train_score /= n_samples
            
            self.train_losses.append(train_loss_not_reg)
            self.train_losses_reg.append(train_loss)
            self.train_scores.append(train_score)

            #-----display training progress-----
            if self.verbose:
                if self.early_stopping or Y_val is not None:
                    print('epoch %d/%d   train loss=%.6f     train score=%.6f     val loss=%.6f    val score=%.6f' 
                        % (epoch+1, self.epochs, train_loss_not_reg, train_score, val_loss, val_score))
                else:
                    print('epoch %d/%d   train error=%.6f' 
                        % (epoch+1, self.epochs, train_loss_not_reg))
            #-----------------------------------

            #-----stopping-----
            self._update_no_improvement_count(epoch, self.train_losses_reg, self.train_scores, self.val_scores)

            if self.no_improvement_count >= self.stopping_patience: # stopping criteria satisfied
                self.set_weights(self.best_weights, self.best_bias)
                break # jump out the for loop
            #------------------
    
    def predict(self, X):
        """
        Make predictions for the given data.

        Parameters:
            - X (np.array): the data to make predictions for.

        Returns:
            - predictions (np.array): the predictions made by the network.
        """
        if self.first_fit:
            raise ValueError("fit has not been called yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if self.layers[0].fan_in != X.shape[1]:
            raise ValueError("X has a different number of features "
                "from the one of the dataset the net has been trained on.")
        
        predictions = self._predict_outputs(X)
        if self.classification:
            predictions = self._outputs_to_labels(predictions)
        return predictions

    def score(self, X_test, Y_test, evaluation_metrics):
        """
        Evaluate the performance of the neural network on the given test data.

        Parameters:
            - X_test (np.array): the test data to evaluate the network on.
            
            - Y_test (np.array): the true target data for the test set.
            
            - evaluation_metrics (str): a list of the evaluation metrics to use.
                Choices: 'mse', 'accuracy'

        Returns:
            - scores (float): the score of the network on the test data according to the chosen evaluation metrics.
        """
        if self.first_fit:
            raise ValueError("fit has not been called yet.")
        if X_test.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if self.layers[0].fan_in != X_test.shape[1]:
            raise ValueError("X has a different number of features "
                "from the one of the dataset the net has been trained on.")
        for evaluation_metric in evaluation_metrics:
            if evaluation_metric not in EVALUATION_METRICS:
                raise ValueError("Unrecognized evaluation metric %s. "
                    "Supported evaluation metrics are %s."% (evaluation_metric, list(EVALUATION_METRICS)))
        if 'accuracy' in evaluation_metrics and self.classification == False:
            raise ValueError("accuracy metric can be used only for classification tasks.")
        if self.classification:
            labels = unique_labels(Y_test)
            for label in labels:
                if label not in self.labels:
                    raise ValueError("test label are not included in train labels.")  
        
        if self.classification == True:
            Y_test = self.binarizer.transform(Y_test).astype(np.float64)
            if self.n_classes == 2 and self.activation_out == 'softmax':
                Y_test = np.hstack((Y_test, 1 - Y_test))

        outputs = self._predict_outputs(X_test)
        if self.classification:
            self.preds = self._outputs_to_labels(outputs)
        else: self.preds = outputs
        
        scores = {}
        for evaluation_metric in evaluation_metrics:
            metric_value = self._evaluate(
                Y_true=Y_test,
                Y_pred=outputs,
                metric=evaluation_metric 
            )
            scores[evaluation_metric] = metric_value
        return scores

    def get_init_weights(self):
        """
        Get the initial weights and bias of the neural network.

        Returns:
            - init_weights (list of np.array): a list of the initial weights of each layer of the network.

            - init_bias (list of np.array): a list of the initial bias of each layer of the network.
        """
        init_weights = []
        init_bias = []
        for layer in self.layers:
            init_weights.append(layer.init_weights)
            init_bias.append(layer.init_bias)
        return init_weights, init_bias

    def set_weights(self, weights, bias):
        """
        Set the weights of the neural network.

        Parameters:
            - weights (list of np.array): a list of the new weights of each layer of the network.
            
            - bias (list of np.array): a list of the new biases of each layer of the network.
        """
        for l, weights_l, bias_l in zip(self.layers, weights, bias):
            l.set_weights(weights_l, bias_l)

    def get_current_weights(self):
        """
        Get the current weights of the neural network.

        Returns:
            - weights (list of np.array): a list of the current weights of each layer of the network.
            
            - bias (list of np.array): a list of the current biases of each layer of the network.
        """
        weights = []
        bias = []
        for l in self.layers:
            weights.append(l.weights)
            bias.append(l.bias)
        return weights, bias

    def set_reinit_weights(self, value: bool):
        self.reinit_weights = value