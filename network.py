import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import unison_shuffle
from math import floor, ceil
from utils import *
from layer import *
from copy import deepcopy
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
        epochs : int = 200,
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
        validation_size : float = 0.1, 
        tol : float = 0.0005,
        validation_frequency : int = 4,
        random_state = None, # TODO: check
        reinit_weights : bool = True # TODO: check
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
        self.evaluation_metric = EVALUATION_METRICS[evaluation_metric]
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
        self.random_state = random_state #TODO: check
        self.reinit_weights = reinit_weights
        self.n_targets = None

    def check_params(self, params):
        if (params['activation_out'] not in ACTIVATIONS):
            raise ValueError("Unrecognized activation_out '%s'. "
                "Supported activation functions are %s." % (params['activation_out'], list(ACTIVATIONS)))
        if (params['activation_hidden'] not in ACTIVATIONS):
            raise ValueError("Unrecognize activation_hidden '%s'. "
                "Supported activation functions are %s."% (params['activation_hidden'], list(ACTIVATIONS)))
        if not isinstance(params['hidden_layer_sizes'], list):
            raise ValueError("hidden_layer_sizes must be a list of integers")
        if any(size <= 0 for size in params['hidden_layer_sizes']):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." % params['hidden_layer_sizes'])
        if params['loss'] not in LOSSES:
            raise ValueError("Unrecognized loss.")
        if params['classification'] == False and params['loss'] == 'logloss':
            raise ValueError("Classification == True required to use logloss as loss function")
        if not (params['loss'] == 'logloss' and params['activation_out'] == 'softmax') and \
            not (params['loss'] != 'logloss' and params['activation_out'] != 'softmax'):
            raise ValueError("Softmax activation function and logloss loss function must be used together")
        if params['epochs'] <= 0:
            raise ValueError("epochs must be > 0, got %s. " % params['epochs'])
        if params['evaluation_metric'] not in EVALUATION_METRICS:
            raise ValueError ("Unrecognized evaluation metric '%s'. "
                "Supported evaluation metrics are %s."% (params['evaluation_metric'], list(EVALUATION_METRICS)))
        if params['learning_rate'] not in ["fixed", "linear_decay"]:
            raise ValueError("Unrecognized learning_rate_schedule '%s'. "
            "Supported learning rate schedules are %s." % (params['learning_rate'], ["fixed", "linear_decay"]))
        if params['learning_rate_init'] <= 0.0:
            raise ValueError("learning_rate_init must be > 0, got %s. " % params['learning_rate_init'] )
        if params['learning_rate'] == "linear_decay":
            if params['tau']  <= 0 or params['tau'] > params['epochs']:
                raise ValueError("tau must be > 0 and <= epochs, got %s." % params['tau'] )
        if params['batch_size']  <= 0:
            raise ValueError("batch_size must be > 0, got %s." % params['batch_size'])
        if params['lambd'] < 0.0:
            raise ValueError("lambd must be >= 0, got %s." % params['lambd'])       
        if params['alpha'] > 1 or params['alpha'] < 0:
            raise ValueError("alpha must be >= 0 and <= 1, got %s" % params['alpha'])     
        if not isinstance(params['verbose'], bool):
            raise ValueError("verbose must be a boolean, got %s" % params['verbose'])
        if not isinstance(params['nesterov'], bool):
            raise ValueError("nesterov must be a boolean, got %s" % params['nesterov'])     
        if params['stopping_patience'] > params['epochs'] or params['stopping_patience']  <= 0:
            raise ValueError("patience must be between 1 and max epochs %s, got %s" % (params['epochs'], params['stopping_patience'] ))
        if not isinstance(params['early_stopping'] , bool):
            raise ValueError("ealry stopping must be a boolean, got %s" % params['early_stopping'])      
        if params['validation_size']  > 100 or params['validation_size']  < 0:
            raise ValueError("validation size must be between 0 and 100 (%), got %s" % params['validation_size'] )   
        if params['tol']  < 0 or params['tol']  > 0.5:
            raise ValueError("tolerance must be > 0 and < 0.5, got %s" % params['tol'] )
        if params['validation_frequency'] > params['epochs'] or params['validation_frequency'] <= 0:
            raise ValueError("validation frequency must be between 1 and max epochs %s, got %s" % (params['epochs'], params['validation_frequency']))
        if not isinstance(params['classification'], bool):
            raise ValueError("classification must be a boolean, got %s" % params['classification'])
        # TODO: fare i check che nel caso di classificazione l'output activation function sia softmax/tanh/logistic
        
    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        outputs = self.predict_outputs(X)
        if self.classification:
            labels = self.outputs_to_labels(outputs)
        else:
            labels = outputs
        return labels

    # predict output for given input
    def predict_outputs(self, X):
        if self.n_targets == None:
            raise ValueError("Net has not been fitted yet.")

        Y = np.empty((X.shape[0], 1, self.n_targets))

        # run network over all samples
        for i in range(X.shape[0]):
            # forward propagation
            output = X[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            Y[i] = output

        Y = Y.reshape(Y.shape[0], Y.shape[2])
        return Y

    def outputs_to_labels(self, Y):
        if not self.classification:
            raise ValueError('For regression tasks network\'s'
                'outputs are already labels.')
        
        if self.n_classes == 2 and self.activation_out != 'softmax':
            if self.activation_out == 'tanh':
                threshold = 0
                neg_label = -1.0
            else:
                threshold = 0.5
                neg_label = 0.0
            Y = np.where(Y > threshold, 1.0, neg_label)
        else:
            # TODO: da sistemare
            Y_new = []
            B = np.max(Y, axis=1)
            for i in range(Y.shape[0]):
                Y_new.append(np.where(Y[i] < B[i], neg_label, 1.0))
            Y = np.array(Y_new)
            # TODO: gestire più massimi
        Y = self.binarizer.inverse_transform(Y).astype(np.float64)
        Y = Y.reshape(Y.shape[0], 1)
        return Y

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

    def compose(self, input_size, output_size):
        self.layers = []
        # Add first hidden layer
        self.add(Layer(
            fan_in = input_size,
            fan_out = self.hidden_layer_sizes[0],
            activation = self.activation_hidden
            ))
        # Add further hidden layers
        for i in range(len(self.hidden_layer_sizes)-1):
            self.add(Layer(
                fan_in = self.hidden_layer_sizes[i],
                fan_out = self.hidden_layer_sizes[i+1],
                activation = self.activation_hidden
            ))
        # Add output layer
        self.add(Layer(
            fan_in = self.hidden_layer_sizes[-1],
            fan_out = output_size,
            activation = self.activation_out
        ))
        self.first_fit = False

    def fit(self, X_train, Y_train): 

        # NOTE: ora Y deve essere una matrice. 
        # Prima accettava anche array ad 1 dim ma sicoome predict deve ritornare matrici 
        # il chiamante deve comunque occuparsi delle dimensioni dei target per invocare le metriche,
        # almeno adesso accetta Y nel formato in cui lo predice.
        if len(X_train.shape) != 2:
            raise ValueError("X_train must be a 2-dimensional array")
        if len(Y_train.shape) != 2:
            raise ValueError("Y_train must be a 2-dimensional array")


        if self.activation_out == 'tanh':
            neg_label = -1.0
        else:
            neg_label = 0.0
        self.binarizer = preprocessing.LabelBinarizer(pos_label=1.0, neg_label=neg_label) # TODO: quando viene fatto it più volte??
        self.binarizer.fit(Y_train)
        Y_train = self.binarizer.transform(Y_train).astype(np.float64)
        
        if self.early_stopping:
            if self.classification:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train,
                    Y_train,
                    test_size=self.validation_size,
                    shuffle=True,
                    stratify=Y_train,
                    random_state=self.random_state
                )
            else:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train,
                    Y_train,
                    test_size=self.validation_size,
                    shuffle=True,
                    random_state=self.random_state
                )
        
        if self.batch_size > X_train.shape[0]:
            raise ValueError("batch_size must not be larger than sample size, got %s." % self.batch_size)
        
        # not needed if in layer we use outer
        #X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        # check if current net structure is compatible with the dataset to fit
        if not self.first_fit and (self.layers[0].fan_in == X_train.shape[-1] and \
                            self.layers[-1].fan_out == Y_train.shape[1]):
            # eventually reinitialize weights
            if self.reinit_weights:
                for layer in self.layers:
                    layer.weights_init()
        else:
            if not self.first_fit and self.reinit_weights == False:
                raise ValueError("Cannot use current weights. "
                "Net structure is not compatible with the dataset to fit.")
            if self.classification:
                self.n_classes = len(self.binarizer.classes_)
                if self.n_classes == 2 and self.activation_out != 'softmax':
                    n_output_units = 1
                else:
                    n_output_units = self.n_classes
            else:
                n_output_units = Y_train.shape[1]
            self.compose(X_train.shape[-1], n_output_units) # TODO: in realtà basta cambiare primo e ultimo layer

        self.n_targets = Y_train.shape[1]
        if self.activation_out == 'softmax' and self.n_targets == 1:
            raise ValueError("Two or more output units are required to use softmax as activation function.")
        if (self.loss == LOSSES['mee'] or self.loss == LOSSES['mrmse']) and self.n_targets == 1:
            raise ValueError("More than two output units are required to use the chosen loss function.")

        # divide training set into batches
        if isinstance(self.batch_size, int):
            n_batches = ceil(X_train.shape[0] / self.batch_size)
        else: # assuming it is a float
            n_batches = floor(1 / self.batch_size)

        train_losses = []
        val_errors = []
        train_scores = []
        val_scores = []

        stopping = self.stopping_patience 

        #-----training loop-----
        # loop max-epoch times
        #   for each bacth       
        #       for each item in the batch
        #           compute weights and bias deltas for curr item
        #           accumulate the deltas
        #       end for
        #   adjust weights and bias deltas using accumulated deltas
        # end loop
        
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
                    output = x
                    
                    
                    # forward propagation
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                      
                    # compute loss and evaluation metric (for display)
                    train_loss += self.loss(y_true=y, y_pred=output)
                    reg_term = 0
                    for layer in self.layers:
                        weights = layer.weights.ravel()
                        reg_term += np.dot(weights, weights)
                    train_loss += self.lambd*reg_term

                    if self.evaluation_metric == 'accuracy':
                        pred_label = self.outputs_to_labels(output)
                    else:
                        pred_label = output
                    train_score += self.evaluation_metric(y_true=y, y_pred=pred_label) # TODO: if mse add reg term?
                    
                    # backward propagation
                    delta = self.loss_prime(y_true=y, y_pred=output) # REVIEW: no need add l2 term (in layer update)
                    for layer in reversed(self.layers):
                        delta = layer.backward_propagation(delta)

                # new learning rate
                self.update_learning_rate(epoch)
                # update weights
                for layer in self.layers:
                    # learning_rate e lambd devono essere scelti ipotizzando di avere 1 solo batch
                    # https://arxiv.org/pdf/1206.5533.pdf (come scalare gli iperparametri in base alla dim del batch)
                    layer.update(
                        #learning_rate=self.learning_rate_curr*(X_batch.shape[0]/X_train.shape[0]),
                        learning_rate=self.learning_rate_curr,
                        batch_size=X_batch.shape[0],
                        alpha=self.alpha,
                        lambd=self.lambd,
                        #lambd=self.lambd*(X_batch.shape[0]/X_train.shape[0]),
                        nesterov=self.nesterov
                    )
            
            # REVIEW: di solito si acculumano gli error pattern per pattern, così risparmiamo anche sui tempi di computazione
            #predict_tr = self.predict(X_train)
            #train_score = self.evaluation_metric(Y_train, predict_tr)
            #train_loss = self.loss(y_true=Y_train, y_pred=predict_tr)
            
            #-----validation-----
            if self.early_stopping:
                if (epoch % self.validation_frequency) == 0:
                    pred_outputs = self.predict_outputs(X_val)
                    val_error = self.loss(y_true=Y_val, y_pred=pred_outputs)
                    if self.evaluation_metric == "accuracy":
                        pred_labels = self.outputs_to_labels(pred_outputs)
                    else:
                        pred_labels = pred_outputs
                    evaluation_score = self.evaluation_metric(Y_val, pred_labels)
            
            # average on all samples 
            train_loss /= X_train.shape[0]
            train_score /= X_train.shape[0]
            
            #-----stopping-----
            if epoch == 10: #init
                precedent_error_increased = train_losses[-1] > train_losses[-2] # error_increased cond. at 9th epoch
                peaks_error_function = 0
                max_error = -1 # epoch in which error function has the last relative minimum point  
                min_error = -1 # epoch in which error function has the last relative max point
                start_peaks_epoch = epoch
                weights_to_return, bias_to_return = self.get_current_weights() 

            if epoch >= 10: # TODO: valutare se incrementare (minimo 30 epoche se errore cresce sempre)
                if self.early_stopping:
                        error_below_tol = val_error <= self.tol
                        rel_error_decrease = (val_errors[-1] - val_error) / val_errors[-1]
                        error_increased = val_error > val_errors[-1]
                        # REVIEW: loss deve essere tale che valore minore => migliore
                else:
                    error_below_tol = train_loss <= self.tol
                    rel_error_decrease = (train_losses[-1] - train_loss) / train_losses[-1]
                    error_increased = train_loss > train_losses[-1]
                    # REVIEW: tipicamente l'errore di training non cresce, settare a False? (attenzione decr)
                   
                if (train_loss > train_losses[-1]) and not precedent_error_increased: # in previous iteration error function was in min
                    min_error = epoch
                    if (min_error - max_error) <= 10: #TODO: parametrico? 
                        if (train_losses[max_error] - train_loss) > 2*self.tol:
                            peaks_error_function += 1
                        else: 
                            peaks_error_function = 0 
                            start_peaks_epoch = epoch
                            weights_to_return, bias_to_return = self.get_current_weights()
                elif not (train_loss > train_losses[-1]) and precedent_error_increased: # in previous iteration error function was in max
                    max_error = epoch
                    if (max_error - min_error) <= 10:
                        if (train_loss - train_losses[min_error]) > 2*self.tol:
                            peaks_error_function += 1   
                        else: 
                            peaks_error_function = 0 
                            start_peaks_epoch = epoch
                            weights_before_peaks, bias_before_peaks = self.get_current_weights()                    
                precedent_error_increased =  train_loss > train_losses[-1]

                if error_below_tol: # if we've already converged (error near 0)
                    stopping = -1
                elif peaks_error_function == 8: # error function is instable
                    stopping = -2 
                elif error_increased or rel_error_decrease < 0.1/100: # if no more significant error decreasing (less than 0.1%) or we are not converging                   
                    stopping -= 1
                else:
                    stopping = self.stopping_patience
                    weights_to_return, bias_to_return = self.get_current_weights()  
                    #self.backtracked_network = deepcopy(self) # keeps track of the best model before early stopping (increasing error)

            
            train_losses.append(train_loss)
            train_scores.append(train_score)

            if self.early_stopping:
                val_errors.append(val_error)
                val_scores.append(evaluation_score)
            
            if self.verbose:
                if self.early_stopping:
                    print('epoch %d/%d   train error=%f     val error=%f    score=%f' 
                        % (epoch+1, self.epochs, train_loss, val_error, evaluation_score))
                else:
                    print('epoch %d/%d   train error=%f' 
                        % (epoch+1, self.epochs, train_loss))
            
            if stopping <= 0: # stopping criteria satisfied
                if stopping == 0: # error function is increasing              
                    if self.early_stopping: # remove values ​​that do not satisfy the criteria
                        val_errors[-self.stopping_patience:] = []
                        val_scores[-self.stopping_patience:] = []
                    train_losses[-self.stopping_patience:] = []
                    train_scores[-self.stopping_patience:] = []
                    self.set_weights(weights_to_return, bias_to_return)
                if stopping == -2: # error function is instable
                    surplus = len(train_losses) - start_peaks_epoch # remove values ​​that do not satisfy the criteria
                    if self.early_stopping:
                        val_errors[-surplus:] = []
                        val_scores[-surplus:] = []
                    train_losses[-surplus:] = []
                    train_scores[-surplus:] = []
                    weights_to_return = weights_before_peaks # set wheights and bias of the last iteration
                    bias_to_return = bias_before_peaks
                    self.set_weights(weights_to_return, bias_to_return)
                break

        if self.early_stopping:
            return train_losses, val_errors, train_scores, val_scores
        else:
            return train_losses, train_scores
        # TODO: return weights_to_return, bias_to_return

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