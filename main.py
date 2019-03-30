import os
import numpy as np
from read_write_features import read_features
from read_write_features import write_features
from read_write_features import read_labels
from read_write_features import write_labels
from feature_extraction import get_features
from feature_extraction import get_labels
from preprocess import train_normalize
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster.k_means_ import KMeans


#K-means algorithm
'''
path_train = "./Train"
num_classes = 6
X = np.zeros((len(os.listdir(path_train)),299*13))

f = open("./X.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)
    X[i] = value
    i += 1

x_train_mean, x_train_std = train_normalize(X)

X = (X - (x_train_mean)) / (x_train_std)
y = get_labels("./Train")

kmeans = KMeans(n_clusters=num_classes, init='k-means++', n_init=10, max_iter = 700, tol = 1e-6)
kmeans.fit(X, y)

path_test = "./Test"
X_test = np.zeros((len(os.listdir(path_test)), 299*13))

f = open("./X_test.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)

    X_test[i] = value
    i += 1

X_test = (X_test - (x_train_mean)) / (x_train_std)

y_test = get_labels(path_test)
y_pred = kmeans.predict(X_test)

print(y_pred + 3)
print(y_test)
print("Test accuracy: ", np.mean((y_pred + 3) == y_test)*100, "%", sep = '')
'''

def rbf(path_train, path_test, pca_comp=200, rbf_gamma=0.0003, rbf_C=10, X = None, X_test = None, y = None, y_test = None):
    '''
    This function trains and tests an svm with rbf kernel according to the data files
    specified in path_train and path_test
    :param path_train: str: path to folder with train files
    :param path_test: str: path to folder with test files
    :param pca_comp: int: number of components that pca will reduce the feature dimensions to
    :param rbf_gamma: float: gamma parameter for rbf kernel
    :param rbf_C: float: C parameters for rbf kernel
    :param X: ndarray (N,D): feature matrix to train rbf kernel with; default is None
    :param X_test: ndarray (number of test samples, D): test feature matrix to test rbf kernel; default is None
    :param y: ndarray (N,): labels for training samples; default is None
    :param y_test: ndarray (number of test samples, ): labels for test samples; default is None
    '''

    print("__________RBF__________")
    pca = PCA(pca_comp)
    rbf = SVC(gamma= rbf_gamma, C= rbf_C, kernel='rbf', class_weight='balanced')

    print("Extracting features for training...")

    if X is None:
        # extract mfcc features for training
        X = get_features(path_train)

    # normalize the features
    x_train_mean, x_train_std = train_normalize(X)
    X = (X - x_train_mean) / x_train_std
    #reduce to pca_comp dimensions using pca
    X = pca.fit_transform(X)

    if y is None:
        y = get_labels(path_train)

    print("\nTraining with RBF...\n")

    # we train using an svm with an rbf kernel, with class weights "balanced"
    rbf.fit(X, y)
    y_pred_train = rbf.predict(X)

    print("Extracting features for testing...")

    if X_test is None:
        X_test = get_features(path_test)

    # apply same normalization and pca dimensionality reduction to test feature matrix
    X_test = (X_test - (x_train_mean)) / (x_train_std)
    X_test = pca.transform(X_test)
    print(X_test.shape)
    print("\nTesting with RBF...\n")

    if y_test is None:
        y_test = get_labels(path_test)
    y_pred = rbf.predict(X_test)

    print("Train accuracy: ", np.mean(y_pred_train == y) * 100, "%", sep='')
    print("Test accuracy: ", np.mean(y_pred == y_test) * 100, "%", sep='')


def linear_svm(path_train, path_test, svm_alpha=0.0005, lr=1e-5, num_iter=300, X = None, X_test = None, y=None, y_test=None):
    '''
    This function trains and tests a linear svm according to the data files
    :param path_train: str: path to folder with train files
    :param path_test: str: path to folder with test files
    :param svm_alpha: float: regularization parameter for linear svm model
    :param lr: float: learning rate parameters for svm model
    :param num_iter: maximum number of the linear svm model
    :param X: ndarray (N,D): feature matrix to train rbf kernel with; default is None
    :param X_test: ndarray (number of test samples, D): test feature matrix to test rbf kernel; default is None
    :param y: ndarray (N,): labels for training samples; default is None
    :param y_test: ndarray (number of test samples, ): labels for test samples; default is None
    '''

    print("__________Linear SVM__________")
    num_classes = 6
    svm = sk.linear_model.SGDClassifier(loss='hinge', penalty='l2',
                                        alpha=svm_alpha, learning_rate='constant', eta0=lr,
                                        max_iter=num_iter, early_stopping=True, class_weight='balanced')

    print("Extracting features for training...")

    if X is None:
        # extract mfcc features for training
        X = get_features(path_train)

    # normalize the features
    x_train_mean, x_train_std = train_normalize(X)
    X = (X - x_train_mean) / x_train_std

    if y is None:
        y = get_labels(path_train)

    print("\nTraining with Linear SVM...\n")

    # initialize W and b to small random weights
    W = np.random.uniform(-0.001, 0.001, (num_classes, 299 * 13))
    b = np.random.uniform(-0.001, 0.001, (num_classes))
    # we train using a linear svm
    svm.fit(X, y, W, b)
    y_pred_train = rbf.predict(X)

    print("Extracting features for testing...")

    if X_test is None:
        X_test = get_features(path_test)

    # apply same normalization to test feature matrix
    X_test = (X_test - (x_train_mean)) / (x_train_std)

    print("\nTesting with Linear SVM...\n")

    if y_test is None:
        y_test = get_labels(path_test)
    y_pred = rbf.predict(X_test)

    print("Train accuracy: ", np.mean(y_pred_train == y) * 100, "%", sep='')
    print("Test accuracy: ", np.mean(y_pred == y_test) * 100, "%", sep='')

def main():
    path_train = "./Train"
    path_test = "./Test"

    N = len(os.listdir(path_train))
    # this number can be determined algebraically: (0.025 + (0.01)(x - 1) = 3 (seconds)); D = x * 13 (cepstrums)
    D = 299 * 13

    X, N, D = get_features(path_train)
    X_test, N_test, _  = get_features(path_test)
    y = get_labels(path_train)
    y_test = get_labels(path_test)
    write_features("./X.txt", X, N, D)
    write_features("./X_test.txt", X_test, N_test, D)
    write_labels("./y.txt",y,N)
    write_labels("./y_test.txt", y_test,N_test)
    X = read_features("./X.txt", N, D)
    X_test = read_features("./X_test.txt", N_test, D)
    y = read_labels("./y.txt", N)
    y_test = read_labels("./y_test.txt", N_test)

    # run on rbf kernel svm model
    rbf(path_train, path_test, X=X, X_test=X_test, y=y, y_test=y_test)
    # run on linear svm model
    linear_svm(path_train, path_test, X=X, X_test=X_test, y=y, y_test=y_test)

if __name__ == '__main__':
    main()