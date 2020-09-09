from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU

dropout = 0.25
filter1 = 32
filter2 = 2*filter1


def createModel(n_classes, input_shape, kernel_size=(3, 3), pool_size=(2, 2), padding='same'):
    model = Sequential(name='AI for CIFAR-10')

    model.add(Conv2D(filter1, kernel_size, padding=padding, activation='relu', input_shape=input_shape, name='CONV11'))
    model.add(Conv2D(filter1, kernel_size, activation='relu', name='CONV12'))
    model.add(MaxPooling2D(pool_size=pool_size, name='MPooling1'))
    model.add(Dropout(dropout, name='DROP1'))

    model.add(Conv2D(filter2, kernel_size, padding=padding, activation='relu', name='CONV21'))
    model.add(Conv2D(filter2, kernel_size, activation='relu', name='CONV22'))
    model.add(MaxPooling2D(pool_size=pool_size, name='MPooling2'))
    model.add(Dropout(dropout, name='DROP2'))

    model.add(Conv2D(filter2, kernel_size, padding=padding, activation='relu', name='CONV31'))
    model.add(Conv2D(filter2, kernel_size, activation='relu', name='CONV32'))
    model.add(MaxPooling2D(pool_size=pool_size, name='MPooling3'))
    model.add(Dropout(dropout, name='DROP3'))

    model.add(Flatten(name='FLATTEN'))

    model.add(Dense(512, activation='relu'))
    model.add(LeakyReLU(name='LEAKY5'))
    model.add(Dropout(dropout*2, name='DROP-last'))
    model.add(Dense(n_classes, activation='softmax', name="OUTPUT"))

    return model
