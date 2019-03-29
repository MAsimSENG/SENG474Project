#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa as li
import numpy as np
import os
            
            
def add_noise( data):
    
        noise = np.random.randn(len(data))
        
       # linearly interpolate noise: (1- alpha) * cur_val + (alpha) * noise
       
        data_noise = (1-0.005) * data + 0.005 * noise

        return data_noise

def shift( data):
            
        return np.roll(data, 20000)



def stretch( data, rate=0.25):
    
        input_length = len(data) # corresponding to 2.5 seconds 
                
        data = li.effects.time_stretch(data, rate)
        
        stretched_length = len(data) # (2.5w)
        
        start_index = int((stretched_length - input_length)/2)
        
        end_index = int((stretched_length + input_length)/2)
        
        data = data[start_index:end_index]
               
        return data


def augmentData(path_to_folder):
    
        data_files = os.listdir(path_to_folder)
         
        for f in data_files:    
            
            time_series, sample_rate = li.core.load(path_to_folder+ "/" + f, sr=16000)
            
            # create copies of the original audio data 

            time_series_addnoise = time_series[:]
            
            time_series_shift = time_series[:]
            
            time_series_stretch = time_series[:]

            # call functions to augment the data 
            
            noisy = add_noise(time_series_addnoise)
            
            shifted = shift(time_series_shift)
            
            stretched = stretch(time_series_stretch)
            
            # create wav files for the augmented data 
            li.output.write_wav(path_to_folder+ "/" +f+'add_noise.wav', noisy, sample_rate)
            
            li.output.write_wav(path_to_folder+ "/" +f+'_shifted.wav', shifted, sample_rate)
            
            li.output.write_wav(path_to_folder+ "/" +f+'_stretched.wav', stretched, sample_rate)
            
    
