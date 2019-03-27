import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

# 03-02-05-02-01-01-11.wav

spf = wave.open('./Test/03-01-03-01-01-01-11.wav','r') #(0.5 to 3 seconds)
#spf = wave.open('./Train/03-02-05-02-01-01-11.wav','r') (1 seconds to 3.5 seconds)

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()


Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(Time,signal)
plt.show()