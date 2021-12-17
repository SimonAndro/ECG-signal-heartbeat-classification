import os
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0'
mit_bir_dest = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0-aami_annotations'


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
DS1 = [101, 106, 108,109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,230]
# records for data set two (test model)
DS2 = [100, 103, 105,111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

record_names = DS1

for record_name in record_names:
    fname = os.path.join(mit_bih_dir, str(record_name))

    try:  # some numbers don't exist in the records
        annotation = wfdb.rdann(fname, 'atr')
        record = wfdb.rdrecord(fname)
    except:
        continue

    symbols = np.array(annotation.symbol)
    samples = np.array(annotation.sample)
    p_signal = record.p_signal    # extract signal

    # test plot
    # plt.plot(p_signal[63315-5000:63315+500,0])
    # plt.show()

    for cat,syms in heartbeat_classes:  # loop over categories
        for s in syms: # loop through the symbols
            ndx = np.nonzero(symbols == s)[0]
            if ndx.size: # indice array isn't empty
                symbols[ndx] = cat

    label, total = np.unique(symbols, return_counts=True)
    print(label, total)
    
    ## extract individual beats
    Rpeak_left = 100
    Rpeak_right = 247

    # pad signal incase of array overflow on extracting beat
    p_signal = p_signal[:,0] # ML II
    
    # normalize between 0 and 1
    p_signal = preprocessing.minmax_scale(p_signal, feature_range=(0, 1), axis=0, copy=True)    

    # loop over beat annotations
    i = 0
    for sym in symbols:
        if sym in known_classes: # symbol is in known category

            # find beat start
            beat_start = samples[i] - Rpeak_right
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
            print(p_signal.size)
            padded_signal = np.pad(p_signal,(left_pad, right_pad),mode='edge')
            single_beat = padded_signal[beat_start:beat_end]
            print(p_signal.size)
           
            print((i, sym,samples[i]))
            plt.plot(single_beat)
            plt.draw()
            plt.title('i is'+str(i))
            plt.pause(0.001)

        i = i + 1

    
    break

    # aami_ann = pd.DataFrame({'symbol':symbols,'sample':samples}) # new annotations dataframe
    # aami_ann.to_csv(os.path.join(mit_bir_dest,str(record_name)+'_label.csv')) # write csv file
    # print(record_name)
    
    

