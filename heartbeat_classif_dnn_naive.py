import os
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

from utilities import load_ecg_signal

# force use CPU instead of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
p_signal, labels, samples, ok = load_ecg_signal(fname)
label, total = np.unique(labels, return_counts=True)

# for l, t in zip(label, total):
#     # print(l, t)  # !debug, peak at label and total

# !debug, visualize part of the physical signal
# plt.figure()
# plt.plot(p_signal[:1000, 0])


def make_dataset(record_names, num_sec, fs, non_normal_labels):
    # function for making dataset ignoring non-beats
    # input:
    # record_names - list of patients
    # num_sec = number of seconds to include before and after the beat
    # fs = frequency
    # output:
    #   X_all = signal (nbeats , num_sec * fs columns)
    #   Y_all = binary is abnormal (nbeats, 1)
    #   sym_all = beat annotation symbol (nbeats,1)

    # initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1, num_cols))
    Y_all = np.zeros((1, 1))
    sym_all = []

    # list to keep track of number of beats across patients
    max_rows = []

    for record_name in record_names:

        fname = os.path.join(mit_bih_dir, str(record_name))

        p_signal, labels, samples, ok = load_ecg_signal(fname)

        if not ok:  # ok is false
            continue

        # grab the first signal, the MLII(Modified Lead II)
        p_signal = p_signal[:, 0]

        # make df to exclude the nonbeats
        k_patient = pd.DataFrame({'labels': labels,
                                  'samples': samples})

        k_patient = k_patient.loc[k_patient.labels.isin(
            non_normal_labels + ['N'])]

        X, Y, sym = build_XY(p_signal, k_patient, num_cols, non_normal_labels)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis=0)
        Y_all = np.append(Y_all, Y, axis=0)

    # drop the first zero row
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]

    # check sizes make sense
    assert np.sum(
        max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of X, Y rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of Y, sym rows messed up'
    return X_all, Y_all, sym_all


def build_XY(p_signal, df_ann, num_cols, abnormal):
    # this function builds the X,Y matrices for each beat
    # it also returns the original symbols for Y

    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows, 1))
    sym = []

    # keep track of rows
    max_row = 0
    for atr_sample, atr_sym in zip(df_ann.samples.values, df_ann.labels.values):
        left = max([0, (atr_sample - num_sec*fs)])
        right = min([len(p_signal), (atr_sample + num_sec*fs)])
        x = p_signal[left: right]
        if len(x) == num_cols:
            X[max_row, :] = x
            Y[max_row, :] = int(atr_sym in abnormal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row, :]
    Y = Y[:max_row, :]
    return X, Y, sym


num_sec = 3
fs = 360
X_all, Y_all, label_all = make_dataset(
    record_names,
    num_sec,
    fs,
    non_normal_labels)

# print(X_all[:20], Y_all[:20], label_all[:20])

# split train, and test data
x_train, x_test, y_train, y_test = train_test_split(
    X_all, Y_all,
    test_size=0.33,
    random_state=42)

# define the model
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=x_train.shape[1]))
model.add(layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# model training
history = model.fit(x_train,
                    y_train,
                    validation_split=0.5,
                    batch_size=64,
                    epochs=20,
                    verbose=1
                    )

print(history.history)

# performance plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# model evaluation
model.evaluate(x_test, y_test)


