#Preprocessor methods for the feature matrix

import numpy as np
from sklearn.decomposition import pca

def train_normalize(x_train):
    '''
    This function finds the mean and std along each dimension of the features
    matrix for feature normalization.

    :param x_train: ndarray(N,D): train features matrix
    :return: x_train_mean: ndarray (D,): mean value along each dimension
             x_train_std: ndarray(D,): standard deviation along each dimension
    '''
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)

    return x_train_mean, x_train_std