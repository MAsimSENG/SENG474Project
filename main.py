import os
import librosa as li
from python_speech_features import mfcc
from feature_extraction import get_features
from feature_extraction import get_labels
from preprocess import train_normalize
import time
import sklearn as sk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster.k_means_ import KMeans

# extract the X and X_test matrices into text files
'''
X = get_features("./Train")

f = open("X.txt", 'w')
for i in range(len(os.listdir("./Train"))):
    row = ''
    for j in range(249*13):
        row = row + str(X[i,j]) + '\t'
    f.write(row.rstrip('\t') + '\n')

f.close()

X_test = get_features("./Test")

g = open("X_test.txt", 'w')
for i in range(len(os.listdir("./Test"))):
    row = ''
    for j in range(249*13):
        row = row + str(X_test[i,j]) + '\t'
    g.write(row.rstrip('\t') + '\n')

g.close()
'''


#SVM RBF Kernel Classifier
'''
path_train = "./Train"
X = np.zeros((len(os.listdir(path_train)),249*13))
pca = PCA(150)
svc = SVC( gamma = 0.0004, C = 10, kernel = 'rbf', class_weight='balanced')

f = open("./X.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)

    X[i] = value
    i += 1

x_train_mean, x_train_std = train_normalize(X)
X = (X - (x_train_mean)) / (x_train_std)
X = pca.fit_transform(X)

y = get_labels(path_train)

print("X and y ready for Training")

svc.fit(X,y)
y_pred_train = svc.predict(X)
print("Train accuracy: ", np.mean(y_pred_train == y)*100, "%", sep = '')


X_test = np.zeros((len(os.listdir("./Test")), 249*13))
f = open("./X_test.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)
    X_test[i] = value
    i += 1
X_test = (X_test - (x_train_mean)) / (x_train_std)
X_test = pca.transform(X_test)

y_test = get_labels("./Test")
y_pred = svc.predict(X_test)

print("Test accuracy: ", np.mean(y_pred == y_test)*100, "%", sep = '')
print(confusion_matrix(y_test, y_pred))
'''

#SVM Classifier
'''
path_train = "./Train"
num_classes = 6
X = np.zeros((len(os.listdir(path_train)),249*13))
W = np.random.uniform(-0.001,0.001,(num_classes,249*13))
b = np.random.uniform(-0.001,0.001,(num_classes))

svm = sk.linear_model.SGDClassifier(loss='hinge', penalty='l2',
    alpha = 0.5, learning_rate='constant', eta0=1e-5, max_iter=200, early_stopping = True, class_weight='balanced')

f = open("./X.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)
    X[i] = value
    i += 1

x_train_mean, x_train_std = train_normalize(X)
X = (X - (x_train_mean)) / (x_train_std)
#X = pca.fit_transform(X)

y = get_labels("./Train")

print("X and y ready for Training")

svm.fit(X,y,W,b)
y_pred_train = svm.predict(X)

print("Train accuracy: ", np.mean(y_pred_train == y)*100, "%", sep = '')

path_test = "./Test"
X_test = np.zeros((len(os.listdir(path_test)), 249*13))

f = open("./X_test.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)

    X_test[i] = value
    i += 1

X_test = (X_test - (x_train_mean)) / (x_train_std)

y_test = get_labels(path_test)
y_pred = svm.predict(X_test)

print("Test accuracy: ", np.mean(y_pred == y_test)*100, "%", sep = '')
print(confusion_matrix(y_test, y_pred))
'''''''''

#K-means algorithm
'''
path_train = "./Train"
num_classes = 6
X = np.zeros((len(os.listdir(path_train)),249*13))

f = open("./X.txt", 'r')
i = 0
for line in f:
    value = np.asarray(line.rstrip('\n').split('\t')).astype(np.float)
    X[i] = value
    i += 1

x_train_mean, x_train_std = train_normalize(X)

X = (X - (x_train_mean)) / (x_train_std)
y = get_labels("./Train")

kmeans = KMeans(n_clusters=num_classes, init='k-means++', n_init=250)
kmeans.fit(X, y)

path_test = "./Test"
X_test = np.zeros((len(os.listdir(path_test)), 249*13))

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


