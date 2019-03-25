from python_speech_features import mfcc
import librosa as li
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# realistically, we should have sample_rate as a parameter (and have nfft as 2^k)
# Also it is definitely worth truncating the "zero-amplitude" part of the wav file (it will speed up the algorithm immensely)
# Note: we have 1152 data points total -- this might not be enough. I would suggest data augmentation

# sample at 16000 since this is default for MFCC calculations
time_series, sample_rate = li.core.load('./Audio_Speech_Actors_01-24/Actor_16/03-01-06-01-01-01-16.wav', sr=16000)

print(len(time_series) / sample_rate)

#total time of the wav file
print("Length of clip: ", len(time_series)/sample_rate, "s", sep='')

# note that since we take s_i(n) over the all n (which is nfft)
# then we must have frame size cropped or padded with zeros
# since nfft = 512 and frame_length = 400, then we will have some zero padding
features = mfcc(time_series, sample_rate)

features_1 = features - np.mean(features, axis = 0)

# shape of the feature matrix
# should be floor/ceil(328.83125) by 13 (default number of cepstrums)
print("Shape of output for one sample:", features.shape)

#Visualize the MFCC without mean difference
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(features, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC')

#Visualize the MFCC with mean difference
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(features_1, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title('MFCC')
plt.show()
