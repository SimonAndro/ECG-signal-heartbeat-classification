import os
import shutil
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pywt
from scipy import signal


mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0'
mit_bih_dest = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0-aami_annotations'


# Categories of heartbeats in the MIT-BIH database based on AAMI.
heartbeat_classes = [
('N', ['N','L','R','e','j']), # normal category
('S', ['A','a','J','S']), # Supraventricular ectopic
('V', ['V','E']), # Ventricular ectopic
('F', ['F']), # Fusion
('Q', ['/','f','Q','r','B','n','?']) # Unknown category (any other beats)
]

# known beat classes
known_classes = np.array(['N','S','V','F','Q'])

# records for data set one (classification model)
DS1 = ['101', '106', '108','109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223','230']
# records for data set two (test model)
DS2 = ['100', '103', '105','111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']



def ecg_denoise(ecg,symbol):

    ##
    #remove baselinewander
    ##
    data = []
    for i in range(len(ecg) - 1):
        Y = ecg[i]
        Y = Y.astype(float)
        data.append(Y)

    # Create wavelet object and define parameters
    w = pywt.Wavelet('db5')  # 选用Daubechies8小波

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=8)  # 将信号进行小波分解

    #小波重构
    bw_denoised = pywt.waverec(np.multiply(coeffs, [0, 1, 1, 1, 1,1,1,1,1]).tolist(), 'db8')  # 将信号进行小波重构

    #提取R波
    # 将读取到的annatations的心拍绘制到心电图上
    plt.figure()
    plt.plot(ecg[258840:260280])
    plt.title("orignal signal")

    plt.figure()
    plt.plot(bw_denoised[258840:260280])
    plt.title("after basewander removal")

    ##
    #remove baselinewander
    ##
    b, a = signal.butter(8, 0.1, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    ecg_1 = signal.filtfilt(b, a, bw_denoised)  #dbw_denoised为要过滤的信号
    b, a = signal.butter(8, 0.007, 'highpass')
    denoised_ecg = signal.filtfilt(b, a,ecg_1 ) 

    plt.figure()
    plt.plot(denoised_ecg[258840:260280])
    plt.title("after low and highpass filtering.")

    plt.show()

    return denoised_ecg
        

def normalize(data):      
    data = np.nan_to_num(data)  # removing NaNs and Infs
    data = preprocessing.minmax_scale(p_signal, feature_range=(0, 1), axis=0, copy=True)
    # data = data - np.mean(data)
    # data = data / np.std(data)
    return data
            
for k in range(0,2): # first and second dataset
    if k == 0: 
        record_names = DS1
        ds_dir = 'train_set'
        sample_name = 'train_samples.hdf'
        label_name = 'train_labels.hdf'
    else:
        record_names = DS2
        ds_dir = 'test_set'
        sample_name = 'test_samples.hdf'
        label_name = 'test_labels.hdf'      
    
    save_dir = os.path.join(mit_bih_dest,ds_dir+'/')

    if os.path.exists(save_dir): # delete directory if exists
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for record_name in record_names:

        x_samples = np.array([]) # initialize empty array for beats
        y_labels = np.array([]) # initialize empty array for beat labels

        fname = os.path.join(mit_bih_dir, str(record_name)) # creating file name

        try:  # some numbers don't exist in the records
            annotation = wfdb.rdann(fname, 'atr')
            record = wfdb.rdrecord(fname)
        except:
            continue

        symbols = np.array(annotation.symbol)
        samples = np.array(annotation.sample)
        p_signal = record.p_signal    # extract signal
        ## extract lead two channel
        if record_name == "114": #this record has inverted channels
            p_signal = p_signal[:,1] # MLII      
        else:
            p_signal = p_signal[:,0] # MLII   

        denoised = ecg_denoise(p_signal, symbols)
        p_signal = denoised

        # test plot
        # plt.plot(p_signal[63315-5000:63315+500,0])
        # plt.show()

        for cat,syms in heartbeat_classes:  # loop over categories
            for s in syms: # loop through the symbols
                ndx = np.nonzero(symbols == s)[0]
                if ndx.size: # indice array isn't empty
                    symbols[ndx] = cat

        # label, total = np.unique(symbols, return_counts=True)
        # print(label, total)
        
        ## extract individual beats
        Rpeak_left = 103
        Rpeak_right = 247

        ###pad signal incase of array overflow on extracting beat  
        # normalize between 0 and 1
        p_signal = normalize(p_signal)  

        # loop over beat annotations
        i = 0
        for sym in symbols:
            if sym in known_classes: # symbol is in known category

                # find beat start
                beat_start = samples[i] - Rpeak_left
                left_pad = 0
                if beat_start < 0:               
                    left_pad = -beat_start
                    beat_start = 0

                # find beat end
                beat_end = samples[i] + Rpeak_right
                right_pad = 0
                if beat_end > p_signal.size:                
                    right_pad = beat_end - p_signal.size
                    beat_end = p_signal.size

                # grab single beat                    
                single_beat = p_signal[beat_start:beat_end]
                single_beat = np.pad(single_beat,(left_pad, right_pad),mode='edge')                

                # append to beat array
                if x_samples.size == 0: # first beat
                    x_samples = np.array(single_beat)
                    y_labels = np.array(sym)
                else:                   
                    x_samples = np.vstack((x_samples,single_beat))
                    y_labels = np.append(y_labels,sym)

                #print(x_samples.shape, y_labels.shape)

                #print((i, sym, samples[i]))
                # plt.plot(single_beat)
                # plt.show()
                # plt.title('i is'+str(i))
                # plt.pause(0.001)


            i = i + 1 # increment i
        print(record_name)
        
        # save samples and labels
        np.save(os.path.join(save_dir,record_name+'_samples'), x_samples)  
        np.save(os.path.join(save_dir,record_name+'_labels'), y_labels)  
