import os
from python_speech_features import mfcc
import librosa as li
import numpy as np

def get_features(path):
    '''
    This function extracts mfcc features from each of the wav files located at the specified path
    and returns a feature matrix containing the mfcc features for each sample
    :param path: str: path to directory containing the wav files for feature extraction
    :return: X: ndarray (N,D): matrix of mfcc features for each sample
    '''

    data_files = os.listdir(path)

    N = len(data_files)
    D = 299 * 13  # this number can be determined algebraically: (0.025 + (0.01)(D - 1) = 3 (seconds)) * 13 (cepstrums)

    # feature matrix
    X = np.zeros((N, D))

    sample_num = 0
    for f in data_files:
        # sample at 16000 since this is default for MFCC calculations
        ts, sample_rate = li.core.load(path + "/" + f, sr=16000)
        # clipped audio will be 3 seconds long (empirical measure)
        clipped_ts = np.zeros(3 * sample_rate)

        file_id = f.rstrip(".wav").split("-")

        # we trim the audio files into 3 second clips. We take particular 3 second clips of the audio
        # depending on the type of audio (song/speech).
        if int(file_id[1]) == 1:
            ts = ts[int(0.5 * sample_rate):int(3.5 * sample_rate)]
        else:
            ts = ts[int(1 * sample_rate):int(4 * sample_rate)]

        # in case the file was less than 3.5 or 4 seconds long, we pad with zeros
        clipped_ts[0:len(ts)] = ts

        # each sample's features are flattened to 1 dimension
        mfcc_vec = mfcc(clipped_ts, sample_rate).flatten()

        assert(mfcc_vec.shape[0] == D)

        X[sample_num] = mfcc_vec
        sample_num += 1

        # keep track of feature extraction process
        if sample_num % 10 == 0:
            print(100 * sample_num / N, "% done extraction", sep = '')

    return X

def get_labels(path):
    '''
    :param path: str: path used to specify whether we get labels for train or test set
    :return: y : ndarray (N,): data sample labels
    '''
    data_files = os.listdir(path)

    # data sample labels, the 3rd number (position 2) in the file name corresponds to the sample's emotion label
    y = np.asarray([int(f.rstrip(".wav").split("-")[2]) for f in data_files])

    return y
