
from neural_network import NeuralNetwork
import numpy as np
import scipy.stats as st
import warnings
import matplotlib.pyplot as plt
from validation import k_fold_cross_validation
from utils import LOSSES, EVALUATION_METRICS

class Ensemble:
    """
    Ensemble of Neural Networks

    Attributes:
        - models_params (list of dict): list of dictionaries, where each dictionary 
            contains the parameters for instantiating a neural network.
        
        - n_trials (int): number of trials to run to obtain the ensemble's prediction.
        
        - models  (list): list of the neural network models in the ensemble
    """
    def __init__(self, models_params, n_trials):
        self.models_params = models_params
        self.loss = self.models_params[0]['loss']
        self.evaluation_metric = self.models_params[0]['evaluation_metric']
        self.classification = self.models_params[0]['classification']
        self.n_trials = n_trials
        self.n_models = len(models_params)
        self.fitted = False

        self.max_epoch = 0
        for i in range(len(models_params)):
            self.max_epoch = max(models_params[i]['epochs'], self.max_epoch)

        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []
        
        self.best_train_losses_mean = np.zeros(shape = (self.n_models))
        self.best_val_losses_mean = np.zeros(shape = (self.n_models))
        self.best_train_scores_mean = np.zeros(shape = (self.n_models))
        self.best_val_scores_mean = np.zeros(shape = (self.n_models))
        self.models = []
    
    def fit(self, X_train, y_train, X_test = None, y_test = None):
        """
        Train the ensemble of neural networks for each param's configuration in the model_params list on 
        the provided data
        
        Parameters:
             - X_train (np.array): the training data.

            - Y_train (np.array): the target data for the training set.

            - X_test (np.array): the test data. Optional, if not provided, no validation score will be calculated.

            - Y_test (np.array): the target data for the test set. Optional, if not provided, no validation score 
                will be calculated.
        """
        self.fitted = True
        self.validation_flag = False

        self.train_losses_trials_mean = []
        self.val_losses_trials_mean = []
        self.train_scores_trials_mean = []
        self.val_scores_trials_mean = []
        
        # tensor of losses/scores for each epoch for every trail of every model
        train_losses = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        val_losses = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        train_scores = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        val_scores = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        self.best_epochs = np.full(shape = (self.n_models, self.n_trials), fill_value = np.nan)

        # loop thru ensemble models
        for i in range(self.n_models):
            print(f"{i+1}/{self.n_models}")
            params =self.models_params[i]
            print(params)
            self.models.append([])
            
            # loop thru the number of trials for each configuration
            for j in range(self.n_trials):
                net = NeuralNetwork(**params)
                net.fit(X_train, y_train, X_test, y_test)
                self.models[i].append(net)

                self.best_train_losses_mean[i] += net.train_losses[net.best_epoch]
                self.best_train_scores_mean[i] += net.train_scores[net.best_epoch]

                if ((X_test is not None and y_test is not None) or net.early_stopping == True):
                    self.validation_flag = True
                    self.best_val_losses_mean[i] += net.val_losses[net.best_epoch]
                    self.best_val_scores_mean[i] += net.val_scores[net.best_epoch]

                train_losses[i][j][:len(net.train_losses)] = np.array(net.train_losses)
                val_losses[i][j][:len(net.val_losses)] = np.array(net.val_losses)
                train_scores[i][j][:len(net.train_scores)] = np.array(net.train_scores)
                val_scores[i][j][:len(net.val_scores)] = np.array(net.val_scores)
                self.best_epochs[i][j] = net.best_epoch

            # matrix of average of losses/scores for each model across every epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.train_losses_trials_mean.append(np.nanmean(train_losses[i], axis=0))
                self.val_losses_trials_mean.append(np.nanmean(val_losses[i], axis=0))
                self.train_scores_trials_mean.append(np.nanmean(train_scores[i], axis=0))
                self.val_scores_trials_mean.append(np.nanmean(val_scores[i], axis=0))
        
        # list of average of the best scores/losses resched on each trials for every model
        self.best_train_losses_mean /= self.n_trials
        self.best_val_losses_mean /= self.n_trials
        self.best_train_scores_mean /= self.n_trials
        self.best_val_scores_mean /= self.n_trials

        # list of averege of losess/scores for each epoch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.train_losses_mean = np.nanmean(self.train_losses_trials_mean, axis=0)
            self.val_losses_mean = np.nanmean(self.val_losses_trials_mean, axis=0)
            self.train_scores_mean = np.nanmean(self.train_scores_trials_mean, axis=0)
            self.val_scores_mean = np.nanmean(self.val_scores_trials_mean, axis=0)

        # remove nan values
        for i in range(self.n_models):
            self.train_losses_trials_mean[i] = self.train_losses_trials_mean[i][~np.isnan(self.train_losses_trials_mean[i])]
            self.val_losses_trials_mean[i] = self.val_losses_trials_mean[i][~np.isnan(self.val_losses_trials_mean[i])]
            self.train_scores_trials_mean[i] = self.train_scores_trials_mean[i][~np.isnan(self.train_scores_trials_mean[i])]
            self.val_scores_trials_mean[i] = self.val_scores_trials_mean[i][~np.isnan(self.val_scores_trials_mean[i])]
            self.train_losses.append([])
            self.val_losses.append([])
            self.train_scores.append([])
            self.val_scores.append([])
            for j in range(self.n_trials):
                self.train_losses[i].append(train_losses[i][j][~np.isnan(train_losses[i][j])])
                self.val_losses[i].append(val_losses[i][j][~np.isnan(val_losses[i][j])])
                self.train_scores[i].append(train_scores[i][j][~np.isnan(train_scores[i][j])])
                self.val_scores[i].append(val_scores[i][j][~np.isnan(val_scores[i][j])])

        self.train_losses_mean =  self.train_losses_mean[~np.isnan(self.train_losses_mean)]
        self.val_losses_mean = self.val_losses_mean[~np.isnan(self.val_losses_mean)]
        self.train_scores_mean = self.train_scores_mean[~np.isnan(self.train_scores_mean)]
        self.val_scores_mean = self.val_scores_mean[~np.isnan(self.val_scores_mean)]

        # ensamble's losses/scores (scalar)
        self.final_train_loss = np.mean(self.best_train_losses_mean)
        self.final_val_loss = np.mean(self.best_val_losses_mean)
        self.final_train_score = np.mean(self.best_train_scores_mean)
        self.finale_val_score = np.mean(self.best_val_scores_mean)

    def predict(self, X):
        """
        Make the ensemble's predictions for the given data.
        
        Parameters:
            - X (np.array): the data to make predictions for.

        Returns:
            - preds (np.array): ensemble's prediction for each sample in X
        """
        if not self.fitted:
            print('Ensemble has not been fitted yet')
            return
        preds = []
        for i in range(self.n_models):
            for j in range(self.n_trials):
                pred = self.models[i][j].predict(X)
                preds.append(pred)

        if self.classification:
            self.preds = st.mode(preds, axis=0).mode
        else: self.preds = np.mean(np.array(preds), axis=0)

        return self.preds
                
    
    def plot(self):
        """
        Plots losses and scores curves of the ensemble. All figures are saved in PDF format.
        """
        if not self.fitted:
            print('Ensemble has not been fitted yet')
            return
        
        # ---------------------- LOSSES ----------------------
        # ------ mean losses ------
        plt.figure()
        plt.semilogy(self.train_losses_mean, color='blue', label='Development Set - Mean', linewidth=1.2)
        if self.validation_flag:
            plt.semilogy(self.val_losses_mean, color='red', linestyle='--', label='Internal Test Set - Mean', linewidth=1.2)
        plt.xlabel('Epochs')
        plt.ylabel(f'Loss ({self.loss.upper()})')
        plt.legend()
        plt.savefig('mean_final_losses.pdf', bbox_inches="tight")
        # -------------------------

        # ------ mean and all models losses ------
        plt.figure()
        label = None
        for i in range(0, self.n_models):
            for j in range(0, self.n_trials):
                if (i==self.n_models-1 and j==self.n_trials-1):
                    label = "Development Set"
                plt.semilogy(self.train_losses[i][j], label = label, color='lightsteelblue', linewidth=0.5)
        if self.validation_flag:
            label = None
            for i in range(0, self.n_models):
                for j in range(0, self.n_trials):
                    if (i==self.n_models-1 and j==self.n_trials-1):
                        label = "Internal Test Set"
                    plt.semilogy(self.val_losses[i][j], label = label, color='lightsalmon', linewidth=0.5) 
        plt.semilogy(self.train_losses_mean, color='blue', label='Development Set - Mean', linewidth=1.2)
        if self.validation_flag: plt.semilogy(self.val_losses_mean, color='red', linestyle='--', label='Internal Test Set - Mean', linewidth=1.2)
        plt.xlabel('Epochs')
        plt.ylabel(f'Loss ({self.loss.upper()})')
        plt.legend()
        plt.savefig('mean_all_models_losses.pdf', bbox_inches="tight")
        # ----------------------------------------

        # ------ plot for each model ------
        for i in range(0, self.n_models):
            plt.figure()
            label = None
            for j in range(0, self.n_trials):
                if (j==self.n_trials-1):
                    label = "Development Set"
                plt.semilogy(self.train_losses[i][j], label = label, color='lightsteelblue', linewidth=0.5)
            if self.validation_flag:
                label = None
                for j in range(0, self.n_trials):
                    if (j==self.n_trials-1):
                        label = "Internal Test Set"
                    plt.semilogy(self.val_losses[i][j], label = label, color='lightsalmon', linewidth=0.5)
            plt.semilogy(self.train_losses_trials_mean[i], color='blue', label=f'Development Set - Mean', linewidth=1.2)
            if self.validation_flag: plt.semilogy(self.val_losses_trials_mean[i], color='red', linestyle='--', label=f'Internal Test Set - Mean', linewidth=1.2)
            plt.xlabel('Epochs')
            plt.ylabel(f'Loss ({self.loss.upper()})')
            plt.legend()
            plt.title(f'Model {i}')
            plt.savefig(f'model_{i}_losses.pdf', bbox_inches="tight")
            # -----------------------------
        # ----------------------------------------------------

        # ---------------------- SCORES ----------------------
        # ------ mean scores ------
        plt.figure()
        plt.semilogy(self.train_scores_mean, color='blue', label='Development Set - Mean', linewidth=1.2)
        if self.validation_flag:
            plt.semilogy(self.val_scores_mean, color='red', linestyle='--', label='Internal Test Set - Mean', linewidth=1.2)  
        plt.xlabel('Epochs')
        plt.ylabel(f'Score ({self.evaluation_metric.upper()})')
        plt.legend()
        plt.savefig('mean_final_scores.pdf', bbox_inches="tight")
        # -------------------------

        # ------ mean and all models scores ------
        plt.figure()         
        label = None
        for i in range(0, self.n_models):
            for j in range(0, self.n_trials):
                if (i==self.n_models-1 and j==self.n_trials-1):
                    label = "Development Set"
                plt.semilogy(self.train_scores[i][j], label = label, color='lightsteelblue', linewidth=0.5)
        if self.validation_flag:
            label = None
            for i in range(0, self.n_models):
                for j in range(0, self.n_trials):
                    if (i==self.n_models-1 and j==self.n_trials-1):
                        label = "Internal Test Set"
                    plt.semilogy(self.val_scores[i][j], label = label, color='lightsalmon', linewidth=0.5) 
        plt.semilogy(self.train_scores_mean, color='blue', label='Development Set - Mean', linewidth=1.2)
        if self.validation_flag: plt.semilogy(self.val_scores_mean, color='red', linestyle='--', label='Internal Test Set - Mean', linewidth=1.2)
        plt.xlabel('Epochs')
        plt.ylabel(f'Score ({self.evaluation_metric.upper()})')
        plt.legend()
        plt.savefig('mean_all_models_scores.pdf', bbox_inches="tight")
        # ----------------------------------------

        # ------ plot for each model ------
        for i in range(0, self.n_models):
            plt.figure()
            label = None
            for j in range(0, self.n_trials):
                if (j==self.n_trials-1):
                    label = "Development Set"
                plt.semilogy(self.train_scores[i][j], label = label, color='lightsteelblue', linewidth=0.5)
            if self.validation_flag:
                label = None
                for j in range(0, self.n_trials):
                    if (j==self.n_trials-1):
                        label = "Internal Test Set"
                    plt.semilogy(self.val_scores[i][j], label = label, color='lightsalmon', linewidth=0.5)
            plt.semilogy(self.train_scores_trials_mean[i], color='blue', label=f'Development Set - Mean', linewidth=1.2)
            if self.validation_flag: plt.semilogy(self.val_scores_trials_mean[i], color='red', linestyle='--', label=f'Internal Test Set - Mean', linewidth=1.2)
            plt.xlabel('Epochs')
            plt.ylabel(f'Score ({self.evaluation_metric.upper()})')
            plt.legend()
            plt.title(f'Model {i}')
            plt.savefig(f'model_{i}_scores.pdf', bbox_inches="tight")
        

    def validate(self, X_train, y_train, k):
        """ 
        Model assessment through a k-fold cross validation for each model of the ensemble.
        
        Parameters: 
            - X_train (np.array): the training data.
        
            - y_train (np.array): the target data for the training set.
        
            - k (int): the number of folds to use for cross validation.

        Returns:
            - results (dict): a dict of the result obtained in the k-fold cross validation process.
        """
        results = [] # list of dictionary returned from k-fold cross validation of each model
        
        for i, params in enumerate(self.models_params):
            print(f"{i+1}/{len(self.models_params)}")
            net = NeuralNetwork(**params)
            net.stopping_criteria_on_loss = False
            result = k_fold_cross_validation(net, X_train, y_train, k, shuffle=False)
            results.append(result)
        
        # tensor contain a matrix for each model contain predictions/losses/scores for each fold
        preds_k = []
        split_train_loss_models = []
        split_train_score_models = []
        # list of averages/scores over all folds for each model
        train_loss_models = []
        train_score_models = []
        
        for i in range(k):
            preds_k.append([])
            split_train_loss_models.append([])
            split_train_score_models.append([])
        for result in results:
            for i, pred in enumerate(result['y_preds']):
                preds_k[i].append(pred)
                split_train_loss_models[i].append(result['split%d_tr_%s'%(i, self.loss)])
                split_train_score_models[i].append(result['split%d_tr_%s'%(i, self.evaluation_metric)])
            train_loss_models.append(result['tr_%s_mean'%self.loss])
            train_score_models.append(result['tr_%s_mean'%self.evaluation_metric])
        
        # average and std deviation of losses/scores across all models
        train_loss_mean = np.mean(train_loss_models)
        train_loss_dev = np.std(train_loss_models)
        train_score_mean = np.mean(train_score_models)
        train_score_dev = np.std(train_score_models)

        y_true_k = results[0]['y_trues'] # true values
        means_pred_k = [] # average of all predictions
        # list of avereges of losses/scores across all models for every split
        pred_split_val_loss = []
        pred_split_val_metric = []
        split_train_loss_mean = []
        split_train_score_mean = []
        
        for i in range(k):
            means_pred_k.append(np.mean(preds_k[i], axis=0))
            pred_split_val_loss.append(LOSSES[self.loss](y_true_k[i], means_pred_k[i]))
            pred_split_val_metric.append(EVALUATION_METRICS[self.evaluation_metric](y_true_k[i], means_pred_k[i]))
            split_train_loss_mean.append(np.mean(split_train_loss_models[i]))
            split_train_score_mean.append(np.mean(split_train_score_models[i]))

        # averege and std deviation of losses/scores across every fold and model
        pred_split_val_loss_mean = np.mean(pred_split_val_loss)
        pred_split_val_loss_dev = np.std(pred_split_val_loss)
        pred_split_val_metric_mean = np.mean(pred_split_val_metric)
        pred_split_val_metric_dev = np.std(pred_split_val_metric)

        cv_results = {
            'tr_%s_mean'%self.loss : train_loss_mean,
            'tr_%s_dev'%self.loss : train_loss_dev,
            'tr_%s_mean'%self.evaluation_metric : train_score_mean,
            'tr_%s_dev'%self.evaluation_metric : train_score_dev,
            'val_%s_mean'%self.loss : pred_split_val_loss_mean,
            'val_%s_dev'%self.loss : pred_split_val_loss_dev,
            'val_%s_mean'%self.evaluation_metric : pred_split_val_metric_mean,
            'val_%s_dev'%self.evaluation_metric : pred_split_val_metric_dev,
        }

        for i in range(k):
            cv_results['split%d_tr_%s'%(i, self.loss)] = split_train_loss_mean[i]
            cv_results['split%d_tr_%s'%(i, self.evaluation_metric)] = split_train_score_mean[i]
            cv_results['split%d_val_%s'%(i, self.loss)] = pred_split_val_loss[i]
            cv_results['split%d_val_%s'%(i, self.evaluation_metric)] = pred_split_val_metric[i]
            cv_results['split%d_best_epoch'%i] = '-'
            
        cv_results['params'] = 'ensemble'
        
        return cv_results