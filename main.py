# import tensorflow as tf
# import theano as t
# import keras as k
#
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
from __future__ import print_function

import matplotlib
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


model1 = createModel(n_classes=nClasses, input_shape=input_shape)

batch_size = 256
epochs = 50

#optimizer=SGD(lr=0.1) #from optimizers
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()

history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_data, test_labels_one_hot))
model1.evaluate(test_data, test_labels_one_hot)


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

predictions = model1.predict(x=test_data, batch_size=batch_size, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_true=test_labels_one_hot, y_pred=rounded_predictions)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels = test_labels_one_hot
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
