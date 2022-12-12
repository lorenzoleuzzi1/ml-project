import matplotlib.pyplot as plt
import numpy as np


def error_plot(tr_error, val_error):
    epochs = len(tr_error)
    epoch_vector = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.plot(epoch_vector, tr_error, "b",
             label="Training error", linewidth=1.5)
    plt.plot(epoch_vector, val_error, "r--",
             label="Validation error", linewidth=1.5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid()
    plt.title("Training and validation error on monks 1 dataset")
    fig_name = "ml-project-ErrorPlot"
    plt.savefig(fig_name)


def accuracy_plot(accuracy):
    epochs = len(accuracy)
    epoch_vector = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.plot(epoch_vector, accuracy, "r",
             linewidth=1.5)
    #plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.title("Validation accuracy on monks 1 dataset")
    fig_name = "ml-project-AccuracyPlot"
    plt.savefig(fig_name)
    