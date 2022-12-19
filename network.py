import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from utils import unison_shuffle
from math import floor, ceil
from utils import *
from layer import *

class Network:
    def __init__(
        self, 
        activation_out,
        activation_hidden='tanh', 
        hidden_layer_sizes=[3], 
        loss='mse', 
        evaluation_metric = 'mse',
        epochs=200,
        learning_rate ='fixed',
        learning_rate_init=0.001,
        tau=100,
        batch_size=1, # 1 = stochastic, 1.0 = un batch
        lambd=0.0001,
        alpha=0.9,
        verbose=True,
        nesterov = False,
        early_stopping_patience = 20, # REVIEW: if low it stops too early
        validation_split = 20,
        tol=0.0005,
        validation_frequency = 4,
        classification=True
        ):

        self.layers = []
        if (activation_out not in ACTIVATIONS):
            raise ValueError("Unrecognized activation_out '%s'. "
                "Supported activation functions are %s." % (activation_out, list(ACTIVATIONS)))
        if (activation_hidden not in ACTIVATIONS):
            raise ValueError("Unrecognize activation_hidden '%s'. "
                "Supported activation functions are %s."% (activation_hidden, list(ACTIVATIONS)))
        self.activation_out = activation_out
        self.activation_hidden = activation_hidden
        if not isinstance(hidden_layer_sizes, list):
            raise ValueError("hidden_layer_sizes must be a list of integers")
        if any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        if loss not in LOSSES:
            raise ValueError("Unrecognized loss.")
        self.loss = LOSSES[loss]
        self.loss_prime = LOSSES_DERIVATIVES[loss]
        if epochs <= 0:
            raise ValueError("epochs must be > 0, got %s. " % epochs)
        
        if evaluation_metric not in EVALUATION_METRICS:
            raise ValueError("Unrecognized evaluation_metric '%s'. "
            "Supported evaluation metrics are %s." % (evaluation_metric, EVALUATION_METRICS))
        self.evalutaion_metric = EVALUATION_METRICS[evaluation_metric]
        
        self.epochs = epochs
        learning_rate_schedules = ["fixed", "linear_decay"]
        if learning_rate not in learning_rate_schedules:
            raise ValueError("Unrecognized learning_rate_schedule '%s'. "
            "Supported learning rate schedules are %s." % (learning_rate, learning_rate_schedules))
        self.learning_rate = learning_rate
        if learning_rate_init <= 0.0:
            raise ValueError("learning_rate_init must be > 0, got %s. " % learning_rate_init)
        self.learning_rate_init = learning_rate_init
        self.learning_rate_curr = learning_rate_init
        self.learning_rate_fin = learning_rate_init * 0.1 # REVIEW: parametrize?
        if tau <= 0 or tau > self.epochs:
            raise ValueError("tau must be > 0 and <= epochs, got %s." % tau)
        self.tau = tau
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0, got %s." % batch_size)
        self.batch_size = batch_size
        if lambd < 0.0:
            raise ValueError("lambd must be >= 0, got %s." % lambd)
        self.lambd = lambd
        if alpha > 1 or alpha < 0:
            raise ValueError("alpha must be >= 0 and <= 1, got %s" % alpha)
        self.alpha = alpha
        self.verbose = verbose

        self.nesterov = nesterov
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        
        self.tol = tol
        self.validation_frequency = validation_frequency
        self.classification = classification

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):

        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        result = np.array(result) # converts external list into numpy array
        if self.y_flatten:
            result = result.reshape(result.shape[0])
        else:
            result = result.reshape(result.shape[0], result.shape[2])
        return result

    def update_learning_rate(self, epoch):
        if self.learning_rate == "fixed":
            self.learning_rate_curr = self.learning_rate_init
        
        if self.learning_rate == "linear_decay":
            if epoch == 0:
                self.learning_rate_curr = self.learning_rate_init
                return 
            if epoch >= self.tau:
                self.learning_rate_curr = self.learning_rate_fin
                return
            
            theta = epoch / self.tau
            lr = (1 - theta) * self.learning_rate_init + theta * self.learning_rate_fin
            
            if (lr < self.learning_rate_fin):
                self.learning_rate_curr = self.learning_rate_fin
            else:
                self.learning_rate_curr = lr

    def fit(self, x_train, y_train):
        if self.classification:
            label = unique_labels(y_train)
            validation_size = max(int(len(x_train)/100 * self.validation_split), len(label))
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, shuffle=True, stratify=y_train)
        else:
            validation_size = max(int(len(x_train)/100 * self.validation_split), 1)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, shuffle=True)
            
        """#shuffle the whole training set 
        x_train, y_train = unison_shuffle(x_train, y_train)
        
        #split training set for validation
        validation_size = max(int(len(x_train)/100 * self.validation_split), 1)
        x_val = x_train[:validation_size]
        y_val = y_train[:validation_size]    
        x_train = x_train[validation_size:]
        y_train = y_train[validation_size:]
        """
        samples = len(x_train)   
        if self.batch_size > samples:
            raise ValueError("batch_size must not be larger than sample size, got %s." % self.batch_size)
        
        #x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1]) # reshape to split

        self.y_flatten = False
        if len(y_train.shape)==1: # TODO: check y_val, y_train have same dim
            self.y_flatten = True
            y_val = y_val.reshape(y_val.shape[0], 1)
            y_train = y_train.reshape(y_train.shape[0], 1)
        
        # Add first hidden layer
        self.add(Layer(
            fan_in=x_train.shape[-1], 
            fan_out=self.hidden_layer_sizes[0], 
            activation=self.activation_hidden#self.activation_hidden, 
            #activation_prime=self.activation_hidden_prime
            ))
        # Add further hidden layers
        for i in range(len(self.hidden_layer_sizes)-1):
            self.add(Layer(
                fan_in=self.hidden_layer_sizes[i], 
                fan_out=self.hidden_layer_sizes[i+1], 
                activation=self.activation_hidden#self.activation_hidden, 
                #activation_prime=self.activation_hidden_prime
            ))
        # Add output layer
        self.add(Layer(
            fan_in=self.hidden_layer_sizes[-1], 
            fan_out=y_train.shape[1], 
            activation=self.activation_out#self.activation_out, 
            #activation_prime=self.activation_out_prime
        ))

        #divide training set into batches
        if isinstance(self.batch_size, int):
            n_batches = ceil(x_train.shape[0] / self.batch_size)
        else: # assuming it is a float
            n_batches = floor(1 / self.batch_size)

        all_train_errors = []
        all_val_errors = []
        all_train_score = []
        all_evalution_scores = []

        stopping = self.early_stopping_patience 


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
            train_error = 0
            x_train, y_train = unison_shuffle(x_train, y_train)
            x_train_batched = np.array_split(x_train, n_batches)
            y_train_batched = np.array_split(y_train, n_batches)

            # for every batches in the set loop  
            for x_batch, y_batch in zip(x_train_batched, y_train_batched):                                      
                # for every patterns in the batch loop
                for x, y in zip(x_batch, y_batch):
                    output = x
                    
                    # forward propagation  
                    for layer in self.layers:
                        output = layer.forward_propagation(output)                        
                      
                    # compute loss (for display)
                    train_error += self.loss(y_true=y, y_pred=output)
                    
                    # backward propagation
                    # REVIEW: rename error --> delta
                    error = self.loss_prime(y_true=y, y_pred=output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error)
                
                #new learning rate
                self.update_learning_rate(epoch)  
                # update (for every batch)
                for layer in self.layers:
                    layer.update(self.learning_rate_curr, x_batch.shape[0], self.alpha, self.lambd, self.nesterov) 
            
            predict_tr = self.predict(x_train)
                
            train_score = self.evalutaion_metric(y_train, predict_tr)
            
            #-----validation-----
            if (epoch % self.validation_frequency) == 0:
                predict_val = self.predict(x_val)
                val_error = self.loss(y_true=y_val, y_pred=predict_val)
                evaluation_score = self.evalutaion_metric(y_val, predict_val)
            
             #-----early stopping-----
            if epoch >= 10:              
                #if we've already converged (validation error near 0)
                if val_error <= self.tol:
                    stopping = 0  
                else:            
                    #if no more significant error decreasing (less than 0.1%) or we are not converging 
                    #all_val_errors[-1] - val_error < val_error/1000 or (removed because small validation)
                    if (all_val_errors[-1] - val_error < val_error/1000 or val_error > all_val_errors[-1]): # TODO: alzare la percentuale
                        stopping -= 1 #decrease the 'patience'
                    else:
                        stopping = self.early_stopping_patience
            
            
            train_error /= samples #average on all samples
            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            all_train_score.append(train_score)
            all_evalution_scores.append(evaluation_score)
            if self.verbose:
              print('epoch %d/%d   train error=%f     val error=%f    score=%f' 
                % (epoch+1, self.epochs, train_error, val_error, evaluation_score))

            if stopping <= 0: break
    
            
        #show stats
        """plt.plot(all_train_errors, label="training", color="blue")
        plt.plot(all_val_errors, label= "validation", color="green")
        plt.plot(all_evalution_scores, label="score",color="red")
        plt.legend(loc="upper right")
        plt.show()"""

        return all_train_errors, all_val_errors, all_train_score, all_evalution_scores
    
    def save(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()

