
from network import Network
import numpy as np
import warnings
import matplotlib.pyplot as plt
from validation import k_fold_cross_validation

class Ensemble:
    def __init__(self, models_params, n_trials):
        self.models_params = models_params
        self.loss = self.models_params[0]['loss']
        self.evaluation_metric = self.models_params[0]['evaluation_metric']
        self.n_trials = n_trials
        self.n_models = len(models_params)

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
        self.preds =[]
 
     
    def fit(self, X_train, y_train, X_test = None, y_test = None):
        self.validation_flag = False

        self.train_losses_trials_mean = []
        self.val_losses_trials_mean = []
        self.train_scores_trials_mean = []
        self.val_scores_trials_mean = []
        
        train_losses = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        val_losses = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        train_scores = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        val_scores = np.full(shape = (self.n_models, self.n_trials, self.max_epoch), fill_value = np.nan)
        self.best_epochs = np.full(shape = (self.n_models, self.n_trials), fill_value = np.nan)

        for i in range(self.n_models):
            print(f"{i}/{self.n_models}")
            params =self.models_params[i]
            print(params)
            self.models.append([])
            
            for j in range(self.n_trials):
                net = Network(**params)   
                net.epochs = 10
                net.fit(X_train, y_train, X_test, y_test)
                self.models[i].append(net)

                self.best_train_losses_mean[i] += net.train_losses[net.best_epoch]               
                self.best_train_scores_mean[i] += net.train_scores[net.best_epoch]

                if (X_test is not None and y_test is not None and net.early_stopping == False):
                    self.validation_flag = True
                    self.best_val_losses_mean[i] += net.val_losses[net.best_epoch]
                    self.best_val_scores_mean[i] += net.val_scores[net.best_epoch]

                train_losses[i][j][:len(net.train_losses)] = np.array(net.train_losses)
                val_losses[i][j][:len(net.val_losses)] = np.array(net.val_losses)
                train_scores[i][j][:len(net.train_scores)] = np.array(net.train_scores)
                val_scores[i][j][:len(net.val_scores)] = np.array(net.val_scores)
                self.best_epochs[i][j] = net.best_epoch


            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.train_losses_trials_mean.append(np.nanmean(train_losses[i], axis=0))
                self.val_losses_trials_mean.append(np.nanmean(val_losses[i], axis=0))
                self.train_scores_trials_mean.append(np.nanmean(train_scores[i], axis=0))
                self.val_scores_trials_mean.append(np.nanmean(val_scores[i], axis=0))
        
        self.best_train_losses_mean /= self.n_trials
        self.best_val_losses_mean /= self.n_trials
        self.best_train_scores_mean /= self.n_trials
        self.best_val_scores_mean /= self.n_trials

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.train_losses_mean = np.nanmean(self.train_losses_trials_mean, axis=0)
            self.val_losses_mean = np.nanmean(self.val_losses_trials_mean, axis=0)
            self.train_scores_mean = np.nanmean(self.train_scores_trials_mean, axis=0)
            self.val_scores_mean = np.nanmean(self.val_scores_trials_mean, axis=0)
        
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

        self.final_train_loss = np.mean(self.best_train_losses_mean)
        self.final_val_loss = np.mean(self.best_val_losses_mean)
        self.final_train_score = np.mean(self.best_train_scores_mean)
        self.finale_val_score = np.mean(self.best_val_scores_mean)

    def predict(self, X_test):
        preds = []

        for i in range(self.n_models):
            for j in range(self.n_trials):
                pred = self.models[i][j].predict(X_test)
                preds.append(pred)
        
        self.preds_mean = np.mean(np.array(preds), axis=0)

        return self.preds_mean 
                
    
    def plot(self):
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
        

    def validate(self, X_train, y_train, k, evaluation_metric):
   
        for network in self.models:
            results = k_fold_cross_validation(network, X_train, y_train, k, ) 
