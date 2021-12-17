import os
import pandas as pd
import wfdb
import numpy as np

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

# other MIT-BIH database labels
nonbeat_labels = ['[', '!'	, ']'	, 'x'	, '('	, ')'	, 'p'	, 't'	, 'u'	, '`',
                  '\'', '^'	, '|'	, '~'	,     '+'	, 's'	, 'T'	, '*'	, 'D'	,
                  '='	, '"', '@'	, ]

# records for data set one (classification model)
DS1 = [101, 106, 108,109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223,230]
# records for data set two (test model)
DS2 = [100, 103, 105,111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

record_names = np.concatenate(DS1,DS2)

for record_name in record_names:
    fname = os.path.join(mit_bih_dir, str(record_name))

    try:  # some numbers don't exist in the records
        annotation = wfdb.rdann(fname, 'atr')
        record = wfdb.rdrecord(fname)
    except:
        continue

    labels = np.array(annotation.symbol)
    samples = np.array(annotation.sample)
    p_signal = record.p_signal    # extract signal

    for cat,syms in heartbeat_classes:  # loop over categories
        for s in syms: # loop through the symbols
            ndx = np.nonzero(labels == s)[0]
            if ndx.size: # indice array isn't empty
                labels[ndx] = cat

    aami_ann = pd.DataFrame({'symbol':labels,'sample':samples}) # new annotations dataframe
    aami_ann.to_csv(os.path.join(mit_bir_dest,str(record_name)+'_label.csv')) # write csv file
    print(record_name)
    



