import os
from python_speech_features import mfcc
import librosa as li
from os import listdir
import numpy as np


def extract_features(path, data_files):
    # number of data samples
    N = len(data_files)
    D = -1 # this number can be determined mathematically

    # feature matrix
    X = np.zeros((N,D))

    sample_num = 0
    for f in data_files():
        '''trim the wav file to three seconds here'''

        # sample at 16000 since this is default for MFCC calculations
        time_series, sample_rate = li.core.load(path + "/" + f, sr=16000)
        #each sample's features are flattened to 1 dimension
        features_vec = mfcc(time_series, sample_rate).flatten()

        '''get the mfcc delta and delta-delta features here'''

        X[sample_num] = features_vec
        sample_num += 1

    return X

# path used to specify whether it is train, validation or test code
def get_features(path):
    data_files = os.listdir(path)

    X = extract_features(path, data_files)

    '''call for feature normalization at this point'''

    '''call for Principal Component Analysis at this point'''

    return X

# path used to specify whether it is train, validation or test code
def get_labels(path):
    data_files = os.listdir(path)
    N = len(data_files)

    # sample labels, the 3rd number (position 2) in the file name corresponds to the sample label
    y = np.asarray([f.rstrip(".wav").split("-")[2] for f in data_files]).reshape((N,1))

    return y

