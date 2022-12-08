import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import ceil, isclose
from utils import *
from layer import *

class Network:
    def __init__(
        self, 
        activation_out,
        activation_hidden='tanh', 
        hidden_layer_sizes=[3], 
        loss='mse', 
        epochs=200,
        learning_rate_schedule = "fixed",
        learning_rate_init=0.001,
        tau=100,
        batch_size=1, 
        lambd=0.0001,
        alpha=0.9
        ):

        self.layers = []
        if (activation_out not in ACTIVATIONS):
            raise ValueError("Unrecognized activation_out '%s'. "
                "Supported activation functions are %s." % (activation_out, list(ACTIVATIONS)))
        if (activation_hidden not in ACTIVATIONS):
            raise ValueError("Unrecognize activation_hidden '%s'. "
                "Supported activation functions are %s."% (activation_hidden, list(ACTIVATIONS)))
        self.activation_out = ACTIVATIONS[activation_out]
        self.activation_out_prime = ACTIVATIONS_DERIVATIVES[activation_out]
        self.activation_hidden = ACTIVATIONS[activation_hidden]
        self.activation_hidden_prime = ACTIVATIONS_DERIVATIVES[activation_hidden]
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
        self.epochs = epochs
        learning_rate_schedules = ["fixed", "linear_decay"]
        if learning_rate_schedule not in learning_rate_schedules:
            raise ValueError("Unrecognized learning_rate_schedule '%s'. "
            "Supported learning rate schedules are %s." % (learning_rate_schedule, learning_rate_schedules))
        self.learning_rate_schedule = learning_rate_schedule
        if learning_rate_init <= 0.0:
            raise ValueError("learning_rate_init must be > 0, got %s. " % learning_rate_init)
        self.learning_rate_init = learning_rate_init
        if tau <= 0 or tau > self.epochs:
            raise ValueError("tau must be > 0 and <= epochs, got %s." % tau)
        self.tau = tau
        if batch_size < 1:
            raise ValueError("batch_size must be >=1, got %s." % batch_size)
        self.batch_size = batch_size
        if lambd < 0.0:
            raise ValueError("lambd must be >= 0, got %s." % lambd)
        self.lambd = lambd
        if alpha > 1 or alpha < 0:
            raise ValueError("alpha must be >= 0 and <= 1, got %s" % alpha)
        self.alpha = alpha


    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # TODO: maybe not strictly needed?
        input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])

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

        #return result
        result = np.array(result) # converts external list into numpy array
        return result.reshape(result.shape[0], result.shape[2]) # come back to 2 dim array

    def update_learning_rate(self, epoch):
        if self.learning_rate_schedule == "fixed":
            self.learning_rate = self.learning_rate_init
        else: # linear decay
            final_learning_rate = self.learning_rate_init * 0.1
            alpha = epoch / self.tau
            learning_rate = (1 - alpha) * self.learning_rate_init + alpha * final_learning_rate
            
            if epoch == 0:
                self.learning_rate = self.learning_rate_init
            
            if (learning_rate < final_learning_rate or epoch >= self.tau):
                self.learning_rate = final_learning_rate
            else:
                self.learning_rate = learning_rate

    def fit(self, x_train, y_train, x_val, y_val):
        # sample dimension first
        samples = len(x_train)   

        if self.batch_size > samples:
            raise ValueError("batch_size must not be larger than sample size, got %s." % self.batch_size)

        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(y_train.shape[0], 1, 1)
        else:
            y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])

        # Add input layer
        self.add(Layer(
            x_train.shape[2], 
            self.hidden_layer_sizes[0], 
            self.activation_hidden, 
            self.activation_hidden_prime
            ))
        # Add hidden layers
        for i in range(len(self.hidden_layer_sizes)-1):
            self.add(Layer(
                self.hidden_layer_sizes[i], 
                self.hidden_layer_sizes[i+1], 
                self.activation_hidden, 
                self.activation_hidden_prime
            ))
        # Add output layer
        self.add(Layer(
            self.hidden_layer_sizes[-1], 
            y_train.shape[1], 
            self.activation_out, 
            self.activation_out_prime
        ))

        #shuffle the whole training set 
        x_train, y_train = unison_shuffle(x_train, y_train)

        #divide training set into batches
        n_batches = ceil(x_train.shape[0] / self.batch_size)
        x_train_batched = np.array_split(x_train, n_batches)
        y_train_batched = np.array_split(y_train, n_batches)
        # x_train = [x_train[i:i + batch_size] for i in range(0, len(x_train), batch_size)]
        # y_train = [y_train[i:i + batch_size] for i in range(0, len(y_train), batch_size)]

        #delta accumulators inizialization
        deltas_weights = []
        deltas_bias = []

        for layer in self.layers:
            deltas_weights.append(np.zeros(shape = layer.weights.shape))
            deltas_bias.append(np.zeros(shape = layer.bias.shape))

        all_train_errors = []
        all_val_errors = []

        #####
        # loop max-epoch times
        #   for each bacth       
        #       for each item in the batch
        #           compute weights and bias deltas for curr item
        #           accumulate the deltas
        #       end for
        #   adjust weights and bias deltas using accumulated deltas
        # end loop
        ######
        stopping = 1000 #param?

        # training loop
        for epoch in range(self.epochs):         
            train_error = 0
            if(self.batch_size == 1):
                x_train_batched, y_train_batched = unison_shuffle(x_train_batched, y_train_batched)

            # for every batches in the set loop  
            for x_batch, y_batch in zip(x_train_batched, y_train_batched):                                      
                
                #shuffle the batch
                x_batch, y_batch = unison_shuffle(x_batch, y_batch)

                # for every patterns in the batch loop
                for x, y in zip(x_batch, y_batch):
                    output = x
                    
                    # forward propagation  
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                      
                    # compute loss (for display)
                    train_error += self.loss(y, output)               
                    
                    # backward propagation
                    error = self.loss_prime(y, output)                    
                    for layer in reversed(self.layers):
                        error, delta_w, delta_b = layer.backward_propagation(error)
                        #accumulate deltas
                        deltas_weights[layer.id] += delta_w
                        deltas_bias[layer.id] += delta_b
                
                #new learning rate
                self.update_learning_rate(epoch)
                #learning_rate = self.learning_rate_fun(epoch)     
                # update (for every batch)
                for layer in self.layers:
                    layer.update(deltas_weights[layer.id], deltas_bias[layer.id], self.learning_rate, 
                        self.batch_size, self.alpha, self.lambd)                  
                    #reset the deltas accumulators
                    deltas_weights[layer.id].fill(0)
                    deltas_bias[layer.id].fill(0)
            
            #-----validation-----
            predict_val = self.predict(x_val)
            # TODO: da sistemare insieme alla parametrizzazione della misura di prestazione (per la regressione f_pred non serve)
            predict_val = f_pred(predict_val)
            val_error = self.loss(y_val, predict_val)
            
            #-----early stopping-----
            if epoch >= 10:              
                #if we've already converged (validation error near 0)
                if val_error <= 0.0005:
                    stopping = 0               
                #if no more significant error decreasing (less than 0.1%) or we are not converging 
                #val_error - all_val_errors[-1] < val_error/100
                print(val_error, all_val_errors[-1])
                if (isclose(val_error, all_val_errors[-1]) or val_error > all_val_errors[-1]): 
                    stopping -= 1 #decrease the 'patience'
                else:
                    stopping = 20
            
            # calculate average error on all samples
            train_error /= samples

            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            print('epoch %d/%d   train error=%f val error=%f' % (epoch+1, self.epochs, train_error, val_error))

            print(stopping)
            if stopping <= 0: break
    
            
        #show the loss plots
        plt.plot(all_train_errors, color="blue")
        plt.plot(all_val_errors, color="green")
        plt.show()

        return all_train_errors, all_val_errors
# TODO: 
    # return (tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size, **kwargs)
    # return tr_error_value, tr_metric_value, val_error_value, val_metric_value
    def save(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()
