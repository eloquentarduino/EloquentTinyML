import numpy as np
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tinymlgen import port

# load and split dataset into train, validation, test
X, y = load_wine(return_X_y=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3)

input_dim = X_train.shape[1:]
output_dim = y.shape[1]

print('input_dim', input_dim)
print('output_dim', output_dim)

# create and train network
# you can customize the layers as you prefer
nn = Sequential()
nn.add(layers.Dense(units=50, activation='relu', input_shape=input_dim))
nn.add(layers.Dense(units=50, activation='relu'))
nn.add(layers.Dense(output_dim, activation='softmax'))

# use categorical_crossentropy for multi-class classification
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, verbose=0)

print('Accuracy: %.1f' % nn.evaluate(X_test, y_test)[1])

# export to file
with open('wine_nn.h', 'w', encoding='utf-8') as file:
    print(port(nn, variable_name='wine_model', pretty_print=True, optimize=False))