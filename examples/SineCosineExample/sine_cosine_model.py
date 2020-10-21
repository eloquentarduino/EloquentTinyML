import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tinymlgen import port

if __name__ == '__main__':
    SAMPLES = 1000
    np.random.seed(1337)
    # generate random x
    X = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES)
    np.random.shuffle(X)
    # use sin and cosine as outputs
    y_sin = np.sin(X)
    y_cos = np.cos(X)
    y = np.vstack((y_sin, y_cos)).T
    # add noise
    y += 0.1 * np.random.randn(*y.shape)
    # split train and test (no validation for this toy example)
    train_size = int(SAMPLES * 0.7)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    # create network
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=300, batch_size=16)
    # predict values
    y_pred = model.predict(X_test)
    print('score', model.evaluate(X_test, y_test))
    # plot predictions vs real
    plt.title('Comparison of predictions and actual values')
    plt.plot(X_test, y_test, 'b.', label='Actual')
    plt.plot(X_test, y_pred, 'r.', label='Predicted')
    plt.legend()
    plt.show()
    # port to Arduino
    c_code = port(model, pretty_print=True)
    print(c_code)