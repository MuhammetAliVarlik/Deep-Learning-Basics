import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
classes = ["airplane", "automobile", "bird", "cat", "deer", "frog", "horse", "ship", "truck"]

Y_train = Y_train.reshape(-1, )
Y_test = Y_test.reshape(-1, )
# Normalize
X_train = X_train / 255
X_test = X_test / 255


def get_model():
    cnn = models.Sequential([
        # cnn
        # padding=valid ,padding=same
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return cnn


with tf.device('/GPU:0'):
    cpu_model = get_model()
    cpu_model.fit(X_train, Y_train, epochs=10)
    cpu_model.evaluate(X_test, Y_test)
