import os
from python_speech_features import mfcc
import librosa as li
import numpy as np

# path used to specify whether it is train, validation or test code
def get_features(path):

    data_files = os.listdir(path)
    '''assume that the data_files are all trimmed'''

    # number of data samples
    N = len(data_files)
    D = 249 * 13  # this number can be determined algebraically: (0.025 + (0.01)(D - 1) = 2.5 (seconds)) * 13 (cepstrums)

    # feature matrix
    X = np.zeros((N, D))

    sample_num = 0
    for f in data_files:
        # sample at 16000 since this is default for MFCC calculations
        time_series, sample_rate = li.core.load(path + "/" + f, sr=16000)

        # we trim the audio files into 3 second clips. We take particular 3 second clips of the audio
        # depending on the type of audio (song/speech).
        file_id = f.rstrip(".wav").split("-")
        if int(file_id[1]) == 1:
            time_series = time_series[int(0.5 * sample_rate):int(3 * sample_rate)]
        else:
            time_series = time_series[int(1 * sample_rate):int(3.5 * sample_rate)]

        # each sample's features are flattened to 1 dimension
        features_vec = mfcc(time_series, sample_rate).flatten()

        assert(features_vec.shape[0] == D)

        '''get the mfcc delta and delta-delta features here'''

        X[sample_num] = features_vec
        sample_num += 1

        # just to keep track of feature extraction process
        if sample_num % 50 == 0:
            print(sample_num)

    return X

# path used to specify whether it is train, validation or test set
def get_labels(path):
    data_files = os.listdir(path)
    N = len(data_files)

    # data sample labels, the 3rd number (position 2) in the file name corresponds to the sample's label
    y = np.asarray([int(f.rstrip(".wav").split("-")[2]) for f in data_files])

    return y

