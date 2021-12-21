import os
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import math
import operator

from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

from utilities import load_ecg_signal

# force use CPU instead of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# path to the test and train datasets
mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0-aami_annotations'
train_ds_dir = os.path.join(mit_bih_dir, "train_set/")
test_ds_dir = os.path.join(mit_bih_dir, "test_set/")

# records for data set one (classification model)
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
       '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
# records for data set two (test model)
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
       '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

# building the network


def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    # end with a softmax layer with 5 units since we have 5 classes to predict
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


num_epochs = 1
all_scores = []
all_trainloss_histories = []
all_valloss_histories = []

classes = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}

# convert label characters to number representation
def char_to_num(data):    

    # print(np.unique(data), "unique classes")
    # print(data.shape, "before char to num")

    for cl in classes:
        data = [classes[cl] if sym == cl else sym for sym in data]

    data = np.array(data, dtype=np.int32)

    # print(np.unique(data), "unique classes")
    # print(data.shape, "after char to num")

    return data

# encodes labels using one-hot encoding
def labels_to_one_hot(labels, dimension=5):
    results = np.zeros((len(labels), dimension))
    for i , label in enumerate(labels):
        results[i, label] = 1
    return results

# test data set
test_data = np.array([])
test_targets = np.array([])
for k in DS2:  # data in test set
    data = np.load(test_ds_dir+k+'_samples.npy')
    targets = char_to_num(np.load(test_ds_dir+k+'_labels.npy'))

    # append to data array
    if test_data.size == 0:  # first beat
        test_data = np.array(data)
        test_targets = np.array(targets)
    else:
        test_data = np.vstack((test_data, data))
        test_targets = np.append(test_targets, targets)

test_targets = labels_to_one_hot(char_to_num(test_targets))

fold = 1
for k in DS1:  # k is the validation fold
    print('processing fold #', fold)
    fold= fold + 1
    val_data = np.load(train_ds_dir+k+'_samples.npy')
    val_targets = labels_to_one_hot(char_to_num(np.load(train_ds_dir+k+'_labels.npy')))

    # print(val_data.shape, val_targets.shape)

    partial_train_data = np.array([])
    partial_train_targets = np.array([])
    for d in DS1:

        if d == k:
            continue  # exclude current fold from train data

        data = np.load(train_ds_dir+d+'_samples.npy')
        targets = np.load(train_ds_dir+d+'_labels.npy')

        # print(np.isnan(data))

        # print(data.shape, targets.shape)

        # append to data array
        if partial_train_data.size == 0:  # first beat
            partial_train_data = np.array(data)
            partial_train_targets = np.array(targets)
        else:
            partial_train_data = np.vstack((partial_train_data, data))
            partial_train_targets = np.append(partial_train_targets, targets)

    # print(partial_train_data.shape, partial_train_targets.shape)

    partial_train_targets = labels_to_one_hot(char_to_num(partial_train_targets))

    # print(np.unique(partial_train_targets), "about to build")
    model = build_model(partial_train_data.shape[1])
    
    # print(model.summary()) #print model summary

    history = model.fit(
        partial_train_data,
        partial_train_targets,
        epochs=num_epochs,
        batch_size=1024,
        verbose=0,
        validation_data=(val_data, val_targets))
    training_loss_history = history.history['loss']
    validation_loss_history = history.history['val_loss']

    all_valloss_histories.append(validation_loss_history)
    all_trainloss_histories.append(training_loss_history)
    val_result = model.evaluate(test_data, test_targets, verbose=0)
    # print(val_result)
    all_scores.append(val_result)

# print(all_scores)

# evaluation of training
epochs = range(1, len(validation_loss_history)+1)

plt.plot(epochs, training_loss_history, 'bo', label='Training loss')
plt.plot(epochs, validation_loss_history, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# model evaluation
results = model.evaluate(test_data, test_targets)
print(f"accuracy on test data:{results[1]}")

# prediction
predictions = model.predict(test_data)

print(predictions[0])

index_max, value_max = max(enumerate(predictions[0]), key=operator.itemgetter(1))

title = "beat is class: "
i = 0
for key in classes:
    if(i == index_max):
        title = title+key+" confidence: "+str(value_max)
        break
    i = i+1

single_beat = test_data[0]
plt.plot(single_beat)
plt.title(title)
plt.show()

