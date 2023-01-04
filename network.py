import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import floor, ceil
from utils import *
from layer import *
from sklearn import preprocessing

class Network:
    def __init__(
        self, 
        activation_out : str,
        classification : bool,
        activation_hidden : str ='tanh', 
        hidden_layer_sizes = [3], 
        loss : str ='mse', 
        evaluation_metric : str = 'mse',
        epochs : int = 200, # TODO: rename with max_epochs
        learning_rate : str = 'fixed',
        learning_rate_init : float = 0.0001,
        tau : int = 200,
        batch_size : int or float = 1, # 1 = stochastic, 1.0 = un batch
        lambd : float = 0.0001,
        alpha : float = 0.9,
        verbose : bool = True,
        nesterov : bool = False,
        early_stopping : bool = True,
        stopping_patience : int = 20, 
        validation_size : int or float = 0.1, # as for batch size
        tol : float = 0.0005,
        validation_frequency : int = 4,
        random_state = None,
        reinit_weights : bool = True,
        weights_dist : str = None,
        weights_bound : float = None,
        ):
       
        self.check_params(locals())
        self.layers = []
        self.first_fit = True
        self.activation_out = activation_out
        self.activation_hidden = activation_hidden
        self.hidden_layer_sizes = hidden_layer_sizes
        self.loss = LOSSES[loss]
        self.loss_prime = LOSSES_DERIVATIVES[loss]
        self.epochs = epochs
        self.evaluation_metric = evaluation_metric
        self.evaluation_metric_fun = EVALUATION_METRICS[evaluation_metric]
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.learning_rate_curr = learning_rate_init
        self.learning_rate_fin = learning_rate_init * 0.1
        self.tau = tau
        self.batch_size = batch_size
        self.lambd = lambd
        self.alpha = alpha
        self.verbose = verbose
        self.nesterov = nesterov
        self.stopping_patience = stopping_patience
        self.early_stopping = early_stopping
        self.validation_size = validation_size
        self.tol = tol
        self.validation_frequency = validation_frequency
        self.classification = classification
        self.random_state = random_state
        self.reinit_weights = reinit_weights
        self.weights_dist = weights_dist # None, 'normal' or 'uniform'
        self.weights_bound = weights_bound # if 'normal' is the std, if 'uniform' in [-weights_bound, weights_bound]
        if self.activation_out == 'tanh': self.neg_label = -1.0
        else: self.neg_label = 0.0
        self.pos_label = 1.0

    def check_params(self, params):
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
        # TODO: per regressione è possibile usare relu? (magari i target sono tutti positivi?)
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
        if params['validation_frequency'] > params['epochs'] or params['validation_frequency'] <= 0:
            raise ValueError("validation_frequency must be between 1 and max epochs %s." % (params['epochs']))
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
        
    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        if self.first_fit:
            raise ValueError("fit has not been called yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if self.layers[0].fan_in != X.shape[1]:
            raise ValueError("X has a different number of features "
                "from the one of the dataset the net has been trained on.")
        
        Y = self.predict_outputs(X)
        if self.classification:
            Y = self.outputs_to_labels(Y)
        return Y

    # predict output for given input
    def predict_outputs(self, X):
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
    
    def discretize_outputs(self, Y):
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

    def outputs_to_labels(self, Y):
        Y_lbl = self.discretize_outputs(Y)
        if self.n_classes == 2 and self.activation_out == 'softmax':
            Y_lbl = Y_lbl[:, 0]
        Y_lbl = self.binarizer.inverse_transform(Y_lbl).astype(np.float64)
        Y_lbl = Y_lbl.reshape(Y_lbl.shape[0], 1)
        return Y_lbl

    def evaluate(self, Y_true, Y_pred):
        if self.evaluation_metric == 'accuracy':
            Y = self.discretize_outputs(Y_pred)
            # TODO: se lasciamo bias con 2 dim occorre fare reshape
        else:
            Y = Y_pred
        return self.evaluation_metric_fun(y_true=Y_true, y_pred=Y)

    def update_learning_rate(self, epoch):
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

    def compose(self):
        self.layers = []
        # Add first hidden layer
        self.add(Layer(
            fan_in = self.n_features,
            fan_out = self.hidden_layer_sizes[0],
            activation = self.activation_hidden,
            weights_dist = self.weights_dist,
            weights_bound = self.weights_bound
            ))
        # Add further hidden layers
        for i in range(len(self.hidden_layer_sizes)-1):
            self.add(Layer(
                fan_in = self.hidden_layer_sizes[i],
                fan_out = self.hidden_layer_sizes[i+1],
                activation = self.activation_hidden,
                weights_dist = self.weights_dist,
                weights_bound = self.weights_bound
            ))
        # Add output layer
        self.add(Layer(
            fan_in = self.hidden_layer_sizes[-1],
            fan_out = self.n_outputs,
            activation = self.activation_out,
            weights_dist = self.weights_dist,
            weights_bound = self.weights_bound
        ))

    def encode_targets(self, Y_train):
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

    def set_fitting(self, X_train, Y_train):
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2-dimensional array")
        if Y_train.ndim != 2:
            raise ValueError("Y_train must be a 2-dimensional array")
        if self.classification and Y_train.shape[1] > 1:
            raise ValueError("Multilabel classification is not supported.")
        if self.batch_size > X_train.shape[0]:
            raise ValueError("batch_size must not be larger than sample size.")

        self.n_features = X_train.shape[1]
        if self.classification:
            Y_train = self.encode_targets(Y_train)
            self.n_outputs = self.n_classes
            if self.n_classes == 2 and self.activation_out != 'softmax':
                self.n_outputs = 1
        else:
            self.n_outputs = Y_train.shape[1]

        if not self.first_fit and not \
            (self.layers[0].fan_in == self.n_features and \
            self.layers[-1].fan_out == self.n_outputs):
            self.first_fit = True
        
        if self.first_fit:
            self.compose()
            self.first_fit = False
        elif self.reinit_weights:
            for layer in self.layers:
                layer.weights_init( self.weights_dist, self.weights_bound)

        return Y_train

    def update_no_improvement_count(self, epoch, train_losses, val_scores):
        if epoch < 10:
            self.best_epoch = epoch
            self.best_metric = val_scores[-1] if self.early_stopping else train_losses[-1]
            self.best_weights, self.best_bias = self.get_current_weights()
            return
        
        if self.early_stopping:
            metric_delta = abs(val_scores[-2] - val_scores[-1]) / val_scores[-2]
            if self.evaluation_metric == 'accuracy':
                converged = val_scores[-1] >= 1-self.tol
                metric_declined = val_scores[-1] < val_scores[-2]
            else:
                converged = val_scores[-1] <= self.tol
                metric_declined = val_scores[-1] > val_scores[-2]
        else:
            metric_delta = (train_losses[-2] - train_losses[-1]) / train_losses[-2]
            converged = train_losses[-1] <= self.tol
            metric_declined = train_losses[-1] > train_losses[-2]

        if converged:
            self.no_improvement_count = self.stopping_patience # if we've already converged (error near 0)
            self.best_epoch = epoch
            self.best_metric = val_scores[-1] if self.early_stopping else train_losses[-1]
            self.best_weights, self.best_bias = self.get_current_weights()
        elif metric_declined or metric_delta < 0.001/100: # TODO: configurabile?
            self.no_improvement_count += 1 # if no more significant error decreasing (less than 0.1%) or we are not converging 
        else:
            self.no_improvement_count = 0
            self.best_epoch = epoch
            self.best_metric = val_scores[-1] if self.early_stopping else train_losses[-1]
            self.best_weights, self.best_bias = self.get_current_weights()

    def fit(self, X_train, Y_train):
        Y_train = self.set_fitting(X_train, Y_train)
        n_samples = X_train.shape[0]
        # if X_train.ndim != 2:
        #     raise ValueError("X_train must be a 2-dimensional array")
        # if Y_train.ndim != 2:
        #     raise ValueError("Y_train must be a 2-dimensional array")
        # if self.classification and Y_train.shape[1] > 1:
        #     raise ValueError("Multilabel classification is not supported.")
        # if self.batch_size > X_train.shape[0]:
        #     raise ValueError("batch_size must not be larger than sample size.")

        # self.n_features = X_train.shape[1]
        # if self.classification:
        #     Y_train = self.encode_targets(Y_train)
        #     self.n_outputs = self.n_classes
        #     if self.n_classes == 2 and self.activation_out != 'softmax':
        #         self.n_outputs = 1
        # else:
        #     self.n_outputs = Y_train.shape[1]

        # if not self.first_fit and not \
        #     (self.layers[0].fan_in == self.n_features and \
        #     self.layers[-1].fan_out == self.n_outputs):
        #     self.first_fit = True
        
        # if self.first_fit:
        #     self.compose()
        #     self.first_fit = False
        # elif self.reinit_weights:
        #     for layer in self.layers:
        #         layer.weights_init( self.weights_dist, self.weights_bound)

        # early stopping validation split
        if self.early_stopping:
            if self.classification:
                stratify = Y_train
            else:
                stratify = None
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
        else: # assuming it is a float
            n_batches = floor(1 / self.batch_size)

        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []

        self.no_improvement_count = 0
 
        for epoch in range(self.epochs):
            train_loss = 0
            train_score = 0
            X_train, Y_train = shuffle(X_train, Y_train, random_state=self.random_state)
            X_train_batched = np.array_split(X_train, n_batches)
            Y_train_batched = np.array_split(Y_train, n_batches)
            # X_train_batched = [X_train[i:i + self.batch_size] for i in range(0, len(X_train), self.batch_size)]
            # Y_train_batched = [Y_train[i:i + self.batch_size] for i in range(0, len(Y_train), self.batch_size)]
            
            # for every batch in the set loop
            for X_batch, Y_batch in zip(X_train_batched, Y_train_batched):
                # for every pattern in the batch loop
                for x, y in zip(X_batch, Y_batch):
                    batch_size = X_batch.shape[0]
                    output = x
                    
                    # forward propagation
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                      
                    # compute loss and evaluation metric (for display)
                    train_loss += self.loss(y_true=y, y_pred=output)
                    train_score += self.evaluate(Y_true=y, Y_pred=output)
                    
                    # backward propagation
                    delta = self.loss_prime(y_true=y, y_pred=output)
                    for layer in reversed(self.layers):
                        delta = layer.backward_propagation(delta)

                # add l2 regularization term to the loss
                reg_term = 0
                for layer in self.layers:
                    weights = layer.weights.ravel()
                    reg_term += np.dot(weights, weights)
                reg_term = self.lambd * reg_term
                train_loss += reg_term

                # new learning rate
                self.update_learning_rate(epoch)
                # update weights
                for layer in self.layers:
                    # learning_rate e lambd devono essere scelti ipotizzando di avere 1 solo batch
                    # https://arxiv.org/pdf/1206.5533.pdf (come scalare gli iperparametri in base alla dim del batch)
                    layer.update(
                        learning_rate=self.learning_rate_curr,
                        batch_size=batch_size,
                        alpha=self.alpha,
                        lambd=self.lambd,
                        nesterov=self.nesterov
                    )
            
            #-----validation-----
            if self.early_stopping and (epoch % self.validation_frequency) == 0:
                Y_val_output = self.predict_outputs(X_val)
                val_loss = self.loss(y_true=Y_val, y_pred=Y_val_output)
                val_score = self.evaluate(Y_true=Y_val, Y_pred=Y_val_output)
                self.val_losses.append(val_loss)
                self.val_scores.append(val_score)
            
            # average on all samples 
            train_loss /= n_samples
            train_score /= n_samples
            self.train_losses.append(train_loss)
            self.train_scores.append(train_score)

            if self.verbose:
                if self.early_stopping:
                    print('epoch %d/%d   train error=%f     val error=%f    score=%f' 
                        % (epoch+1, self.epochs, train_loss, val_loss, val_score))
                else:
                    print('epoch %d/%d   train error=%f' 
                        % (epoch+1, self.epochs, train_loss))
            
            #-----stopping-----
            self.update_no_improvement_count(epoch, self.train_losses, self.val_scores)

            if self.no_improvement_count >= self.stopping_patience: # stopping criteria satisfied
                # TODO: lasciare o stampare una 'x' sul grafico?
                """if self.early_stopping:
                    self.val_losses[-self.stopping_patience:] = []
                    self.val_scores[-self.stopping_patience:] = []
                self.train_losses[-self.stopping_patience:] = [] 
                self.train_scores[-self.stopping_patience:] = []"""
                self.set_weights(self.best_weights, self.best_bias)
                break # jump out the for loop

    def get_init_weights(self):
        init_weights = []
        init_bias = []
        for layer in self.layers:
            init_weights.append(layer.init_weights)
            init_bias.append(layer.init_bias)
        return init_weights, init_bias

    def set_weights(self, weights, bias):
        for l, weights_l, bias_l in zip(self.layers, weights, bias):
            l.set_weights(weights_l, bias_l)

    def get_current_weights(self):
        weights = []
        bias = []
        for l in self.layers:
            weights.append(l.weights)
            bias.append(l.bias)
        return weights, bias

    def set_reinit_weights(self, value: bool):
        self.reinit_weights = value

    def save(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()
