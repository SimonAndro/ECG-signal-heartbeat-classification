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
from keras.utils.vis_utils  import plot_model


from utilities import load_ecg_signal

# force use CPU instead of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# path to the test and train datasets
mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0-aami_annotations/intra'

labels_file = mit_bih_dir+"/beat_labels.npy"
samples_file = mit_bih_dir+"/beat_samples.npy"

data = np.load(samples_file,allow_pickle=True)
targets = np.load(labels_file,allow_pickle=True)

# train test split
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.33,shuffle=True)

#train validation split
train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size=0.2,shuffle=True)

classes = {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}

# test data statistics
label_total = test_targets.sum(axis=0)

label_total = [(c, label_total[classes[c]]) for c in classes]

print("test data statistics", label_total)

# plt.figure()
# plt.plot(train_data[100])
# plt.show()
# plt.figure()

# training globals
num_epochs = 200

# building the network
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(200, activation='tanh',
                           input_shape=(input_shape,)))
    model.add(layers.Dense(50, activation='tanh'))
    # end with a softmax layer with 5 units since we have 5 classes to predict
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

# print(np.unique(partial_train_targets), "about to build")
model = build_model(train_data.shape[1])

print(model.summary()) #print model summary
plot_model(model,"dnn_intra_model.png")

history = model.fit(
    train_data,
    train_targets,
    epochs=num_epochs,
    batch_size=1024,
    verbose=1,
    validation_data=(val_data, val_targets)
    )
training_loss_history = history.history['loss']
validation_loss_history = history.history['val_loss']

# evaluation of training
epochs = range(1, len(training_loss_history)+1)

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

# print(predictions[0])

index_max, value_max = max(enumerate(predictions[0]), key=operator.itemgetter(1))

title = "beat is class: "
i = 0
for key in classes:
    if(i == index_max):
        title = title+key+" confidence: "+str(value_max)+"actual class is "+str(test_targets[0])
        break
    i = i+1

single_beat = test_data[0]
plt.plot(single_beat)
plt.title(title)
plt.show()

