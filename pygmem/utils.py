import os
import numpy as np
import matplotlib.pyplot as plt
import torch


#Plot data function
def plot_data(regression_type, X, y, predictions=None, figure_file=None):
    '''
    Function which plot the data easily (to modify)
    '''
    plt.scatter(X, y, s=80, label="True labels", alpha=0.2)
    if predictions is not None:
        if regression_type == "Classification":
            predictions = np.argmax(predictions, axis=1)
        plt.scatter(X, predictions, s=10, label="Predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("{} data".format(regression_type))
    plt.legend()
    # if figure_file is not None:
    #     plt.savefig(figure_file)
    plt.show()

def weighted_mse_loss(inputs, target):
    weight = (target + 1)/(torch.sum(target + 1))
    # print(weight)
    # print(torch.mean(inputs - target)**2)
    # print(torch.sum(target * ((inputs - target) ** 2 )/scale))
    return torch.sum(weight * ((inputs - target) ** 2 ))