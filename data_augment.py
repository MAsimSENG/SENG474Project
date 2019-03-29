#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:33:44 2019

@author: Asim
"""
import librosa as li
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

def add_noise(data, alpha):
    for i in range(len(data)):
        data[i] = (1-alpha) * data[i] + (alpha) * np.random.randn(1)[0]
    return data

def shift(data):
    return np.roll(data, -10000)

def stretch(data, compress_ratio):
    data = li.effects.time_stretch(data, compress_ratio)
    return data

def plot_time_series(data, title):
    fig = plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 4, len(data)), data)
    plt.show()

def main():
    
    data_files = os.listdir("./somefiles")
    print(data_files)

    for f in data_files[0:1]:
        file_id = f.rstrip('.wav').split('-')

        time_series, sample_rate = li.core.load("./somefiles"+ "/" + f, sr=16000)

        # create copies of the original audio data
        time_series_add_noise = copy.deepcopy(time_series)
        time_series_shift = copy.deepcopy(time_series)
        time_series_stretch = copy.deepcopy(time_series)
        time_series_compress = copy.deepcopy(time_series)

        # call function to augment the data
        noisy = add_noise(time_series_add_noise, 0.005)
        shifted = shift(time_series_shift)
        stretched = stretch(time_series_stretch, 0.9)
        compressed = stretch(time_series_compress, 1.2)

        if int(file_id[1]) == 1:
            noisy = noisy[int(0.25 * sample_rate):int(2.75 * sample_rate)]
            stretched = stretched[int(0.25 * sample_rate):int(2.75 * sample_rate)]
            shifted = shifted[int(0.75 * sample_rate):int(3.25 * sample_rate)]
            compressed = compressed[int(0.25 * sample_rate):int(2.75 * sample_rate)]
        else:
            noisy = noisy[int(1 * sample_rate):int(3.5 * sample_rate)]
            stretched = stretched[int(1 * sample_rate):int(3.5 * sample_rate)]
            shifted = shifted[int(1 * sample_rate):int(3.5 * sample_rate)]
            compressed = compressed[int(1 * sample_rate):int(3.5 * sample_rate)]

        li.output.write_wav("./somefiles/" + f.rstrip(".wav")+'-addnoise.wav', noisy, sample_rate)
        li.output.write_wav("./somefiles/" + f.rstrip(".wav")+'-shifted.wav', shifted, sample_rate)
        li.output.write_wav("./somefiles/" + f.rstrip(".wav")+'-stretched.wav', stretched, sample_rate)
        li.output.write_wav("./somefiles/" + f.rstrip(".wav") + '-compressed.wav', compressed, sample_rate)

        plot_time_series(noisy, "Added Noise")
        plot_time_series(shifted, "Shifted back by 10/16 seconds")
        plot_time_series(stretched, "0.9 times as fast (Stretched)")
        plot_time_series(compressed, "1.2 times as fast (Compressed)")
    
if __name__ == "__main__":
    main()