# import tensorflow as tf
# import theano as t
# import keras as k
#
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.datasets import cifar10

# from models.model import createModel
from models.model import createModel

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print('Training data shape:', train_images.shape, train_labels.shape)
print('Testing data shape:', test_images.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs:', nClasses)

plt.figure(figsize=[4, 2])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_images[1,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[1]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_images[1,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[1]))

plt.show()

### .... ###

nRows, nCols, nDims = train_images.shape[1:]
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)
test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Normalize the data between 0-1
train_data /= 255
test_data /= 255

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

print('Original label[0]:', train_labels[0])
print('To_categorical (one-hot) label[0]:', train_labels_one_hot[0])


# Now create the model!
# def createModel():
#     model = Sequential()
#
#     model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nClasses, activation='softmax'))
#
#     return model

model1 = createModel(n_classes=nClasses, input_shape=input_shape)
# model1 = createModel()
batch_size = 256
epochs = 10
#optimizer=SGD(lr=0.1) #from optimizers
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_data, test_labels_one_hot))
model1.evaluate(test_data, test_labels_one_hot)
