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

#combined dataset
DS = [ '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', 
        '208', '209', '215', '220', '223', '230', '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
        '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']



def normalize(data):      
    data = np.nan_to_num(data)  # removing NaNs and Infs
    data = preprocessing.minmax_scale(p_signal, feature_range=(0, 1), axis=0, copy=True)
    return data


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
        



classes = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}

# convert label characters to number representation
def char_to_num(data):    

    # print(np.unique(data), "unique classes")
    # print(data.shape, "before char to num")

    for cl in classes:
        data = [classes[cl] if sym == cl else sym for sym in data] #replace symbols with number representations

    data = np.array(data, dtype=np.int32)

    # print(np.unique(data), "unique classes")
    # print(data.shape, "after char to num")

    return data

# encodes labels using one-hot encoding
def labels_to_one_hot(labels, dimension=5):
    results = np.zeros((len(labels), dimension), dtype=np.int32)
    for i , label in enumerate(labels):
        results[i, label] = 1
    return results
            

ds_dir = 'intra'  

save_dir = os.path.join(mit_bih_dest,ds_dir+'/')

if os.path.exists(save_dir): # delete directory if exists
    shutil.rmtree(save_dir)
os.makedirs(save_dir)


## points left and right of the Rpeak
Rpeak_left = 103
Rpeak_right = 247

beat_length = Rpeak_left + Rpeak_right
total_samples = 100733 #got after a test run

# preallocate array size
sample_count = 0
x_samples = np.array(total_samples*[beat_length*[0.0]])  #initialize empty array for beats
y_labels = np.array(total_samples*[None]) # initialize empty array for beat labels

for record_name in DS:

    fname = os.path.join(mit_bih_dir, str(record_name)) # creating file name

    try:  # if some numbers don't exist in the records
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

            # # append to beat array
            # if x_samples.size == 0: # first beat
            #     x_samples = np.array(single_beat)
            #     y_labels = np.array(sym)
            # else:              
            #     # x_samples = np.vstack((x_samples,single_beat))
            #     y_labels = np.append(y_labels,sym)

            # insert samples into array
            y_labels[sample_count] = sym
            x_samples[sample_count][:] = single_beat
            sample_count = sample_count + 1

            #print(x_samples.shape, y_labels.shape)

            #print((i, sym, samples[i]))
            # plt.plot(x_samples[sample_count-1][:])
            # #plt.show()
            # plt.title('i is'+str(i))
            # plt.pause(0.001)


        i = i + 1 # increment i
    print(record_name)

y_labels = labels_to_one_hot(char_to_num(y_labels))
print(y_labels)

# save samples and labels
np.save(os.path.join(save_dir,'beat_samples'), x_samples)  
np.save(os.path.join(save_dir,'beat_labels'), y_labels)  

