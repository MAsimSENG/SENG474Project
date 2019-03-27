import os
import librosa as li
from python_speech_features import mfcc
from feature_extraction import get_features
from feature_extraction import get_labels
from preprocess import train_normalize
import time
import sklearn as sk
import numpy as np


num_classes = 3

W = np.random.uniform(-0.001,0.001,(num_classes, 3237))
b = np.random.uniform(-0.001,0.001,(num_classes))

t0 = time.perf_counter()
X = get_features("./Train")
y = get_labels("./Train")

x_train_mean, x_train_std = train_normalize(X)

X = (X - (x_train_mean)) / (x_train_std)

svm = sk.linear_model.SGDClassifier(loss='hinge', penalty='l2', learning_rate='constant', eta0=1e-4)
svm.fit(X,y,W,b)

print("Train time:", time.perf_counter() - t0)

t1 = time.perf_counter()
X_test = get_features("./Test")
y_test = get_labels("./Test")

X_test = (X_test - (x_train_mean)) / (x_train_std)

y_pred = svm.predict(X_test)

print("Test time:", time.perf_counter() - t1)
print("Test accuracy:", np.mean(y_pred == y_test))




