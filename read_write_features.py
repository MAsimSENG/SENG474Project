import numpy as np

def read_features(read_file, N, D):
    '''
    This function reads features from a specified file and creates and returns a feature matrix
    :param read_file: str: file to read features
    :param N: int: number of data samples
    :param D: int: number of dimensions per sample
    :return X: ndarray (N,D): feature matrix
    '''

    X = np.zeros((N, D))

    print("Reading from {}...".format(read_file))

    f = open(read_file, 'r')
    i = 0
    for line in f:
        value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)

        X[i] = value
        i += 1

    return X

def write_features(write_file, X, N, D):
    '''
    This function writes the feature matrix to a tab-separated text file
    :param write_file: str: file to write to
    :param X: ndarray (N,D): feature matrix to write to file
    :param N: int : number of data samples
    :param D: int: dimension of feature space
    '''

    print("Writing to {}...".format(write_file))

    # write extracted features to tab-separated text file
    f = open(write_file, 'w')
    for i in range(N):
        row = ''
        for j in range(D):
            row = row + str(X[i, j]) + '\t'
        f.write(row.rstrip('\t') + '\n')

    f.close()

def read_labels(read_file, N):
    '''
    This function reads labels from a specified file and creates and returns a label array
    :param read_file: str: file to read labels
    :param N: int: number of data samples
    :return y: ndarray (N,): array of labels
    '''

    y = np.zeros(N)

    print("Reading from {}...".format(read_file))

    f = open(read_file, 'r')
    i = 0
    for line in f:
        value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)

        y[i] = value
        i += 1

    return y

def write_labels(write_file, y, N):
    '''
    This function writes the array of labels to a text file
    :param write_file: str: file to write to
    :param y: ndarray (N,): array of labels to write to file
    :param N: int : number of data samples
    '''

    print("Writing to {}...".format(write_file))

    # write labels to text file
    f = open(write_file, 'w')
    for i in range(N):
        f.write(str(y[i]) + '\n')

    f.close()
