import math
import numpy as np
from sklearn.datasets import load_digits
import tensorflow as tf
from tensorflow.keras import layers
from tinymlgen import port


def get_data():
    np.random.seed(1337)
    x_values, y_values = load_digits(return_X_y=True)
    x_values /= x_values.max()
    # reshape to (8 x 8 x 1)
    x_values = x_values.reshape((len(x_values), 8, 8, 1))

    # split into train, validation, test
    TRAIN_SPLIT = int(0.6 * len(x_values))
    TEST_SPLIT = int(0.2 * len(x_values) + TRAIN_SPLIT)
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

    return x_train, x_test, x_validate, y_train, y_test, y_validate

def get_model():
    x_train, x_test, x_validate, y_train, y_test, y_validate = get_data()

    # create a CNN
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(8, 8, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    # model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(len(np.unique(y_train))))

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=16,
              validation_data=(x_validate, y_validate))
    return Exp


def test_model(model, x_test, y_test):
    x_test = (x_test / x_test.max()).reshape((len(x_test), 8, 8, 1))
    y_pred = model.predict(x_test).argmax(axis=1)
    print('ACCURACY', (y_pred == y_test).sum() / len(y_test))
    exit()


if __name__ == '__main__':
    model, x_test, y_test = get_model()
    test_model(model, x_test, y_test)
    c_code = port(model, variable_name='digits_model', pretty_print=True)
    print(c_code)