import os
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np

from utilities import load_ecg_signal

# path to the dataset directory
mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0'

# patient list, 100 to 234
record_names = np.arange(100, 235)

dataframe = pd.DataFrame()

for record_name in record_names:
    fname = os.path.join(mit_bih_dir, str(record_name))

    try:  # some numbers don't exist in the records
        annotation = wfdb.rdann(fname, 'atr')
    except:
        continue

    labels = annotation.symbol

    # a data frame for a single patient
    label, total = np.unique(labels, return_counts=True)
    k_patient = pd.DataFrame(
        {'label': label, 'total': total, 'patient': record_name})

    dataframe = pd.concat([dataframe, k_patient], axis=0)


total_by_label = dataframe.groupby(
    'label').total.sum().sort_values(ascending=False)
# print(total_by_label)  # !debug, peek at the total by label

# normal, abnormal and non-beat labels
non_normal_labels = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E',
                     '/', 'f', 'Q', '?', ]
nonbeat_labels = ['[', '!'	, ']'	, 'x'	, '('	, ')'	, 'p'	, 't'	, 'u'	, '`',
                  '\'', '^'	, '|'	, '~'	,     '+'	, 's'	, 'T'	, '*'	, 'D'	,
                  '='	, '"', '@'	, ]

# -1 for non-heartbeat labels
dataframe['category'] = -1

# 0 for normal heart beat labels
dataframe.loc[dataframe.label == 'N', 'category'] = 0

# 1 for non normal heart beat labels
dataframe.loc[dataframe.label.isin(non_normal_labels), 'category'] = 1

total_by_category = dataframe.groupby('category').total.sum()
# print(total_by_category)  # !debug, peek at the total by category

fname = os.path.join(mit_bih_dir, str(record_names[0]))
p_signal, labels, samples = load_ecg_signal(fname)
label, total = np.unique(labels, return_counts=True)

# for l, t in zip(label, total):
#     # print(l, t)  # !debug, peak at label and total

# !debug, visualize part of the physical signal
# plt.figure()
# plt.plot(p_signal[:1000, 0])
