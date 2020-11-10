import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tinymlgen import port


def get_model():
    SAMPLES = 1000
    np.random.seed(1337)
    x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
    # shuffle and add noise
    np.random.shuffle(x_values)
    y_values = np.vstack((np.sin(x_values), np.cos(x_values))).T
    y_values += 0.1 * np.random.randn(*y_values.shape)

    # split into train, validation, test
    TRAIN_SPLIT =  int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

    # create a NN with 2 layers of 16 neurons
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=100, batch_size=16,
                        validation_data=(x_validate, y_validate))
    return model


def test_model(model, verbose=False):
    x_test = np.random.uniform(low=0, high=2*math.pi, size=100)
    y_test = np.vstack((np.sin(x_test), np.cos(x_test))).T
    y_pred = model.predict(x_test)
    if verbose:
        for i in range(y_pred.shape[1]):
            print('MAE[%d] = %.3f' % (i, np.abs(y_pred[:, i] - y_test[:, i]).mean()))


if __name__ == '__main__':
    model = get_model()
    test_model(model, verbose=True)
    c_code = port(model, pretty_print=True)
    print(c_code)