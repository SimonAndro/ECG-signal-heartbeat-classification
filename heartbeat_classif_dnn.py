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

# path to the test and train datasets
mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0-aami_annotations'
train_ds_dir = os.path.join(mit_bih_dir,"train_set/")
test_ds_dir = os.path.join(mit_bih_dir, "test_set/")

# records for data set one (classification model)
DS1 = ['101', '106', '108','109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223','230']
# records for data set two (test model)
DS2 = ['100', '103', '105','111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# building the network
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mae'])
    return model

num_epochs = 20
all_scores = []
all_mae_histories = []

def char_to_num(data):

    classes = {'F':0,'N':1,'Q':2,'S':3,'V':4}

    print(np.unique(data), "unique classes")
    print(data.shape, "before char to num")

    for cl in classes:
        data = [classes[cl] if sym == cl else sym for sym in data]

    data = np.array(data, dtype=np.int32)
        
    print(np.unique(data), "unique classes")
    print(data.shape, "after char to num")

    return data

for k in DS1: # k is the validation fold
    print('processing fold #', k)
    val_data = np.load(train_ds_dir+k+'_samples.npy')
    val_targets = char_to_num(np.load(train_ds_dir+k+'_labels.npy'))

    # print(val_data.shape, val_targets.shape)

    partial_train_data = np.array([])
    partial_train_targets = np.array([])
    for d in DS1:

        if d==k: continue # exclude current fold from train data

        data = np.load(train_ds_dir+d+'_samples.npy')
        targets = np.load(train_ds_dir+d+'_labels.npy')

        # print(np.isnan(data))

        # print(data.shape, targets.shape)

        # append to beat array
        if partial_train_data.size == 0: # first beat
            partial_train_data = np.array(data)
            partial_train_targets = np.array(targets)
        else:                   
            partial_train_data = np.vstack((partial_train_data,data))
            partial_train_targets = np.append(partial_train_targets,targets)

    print(partial_train_data.shape, partial_train_targets.shape)
    
   
    partial_train_targets = char_to_num(partial_train_targets)
    
    print(np.unique(partial_train_targets), "about to build")
    model = build_model(partial_train_data.shape[1])
    history = model.fit(
                        partial_train_data,
                        partial_train_targets,
                        epochs=num_epochs, 
                        batch_size=512, 
                        verbose=1)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


print(average_mae_history)
smooth_mae_history = smooth_curve(average_mae_history)
print(smooth_mae_history)

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
