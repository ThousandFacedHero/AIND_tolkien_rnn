import numpy as np
import string
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# function below transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[x:window_size+x] for x in range(len(series)-window_size)]
    y = [series[window_size+x] for x in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), name='LSTM_1'))
    model.add(Dense(1, name='Dense_1'))
    return model


# return the text input with only ascii lowercase and the punctuation
# given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    allowed = ''.join(punctuation) + ' ' + string.ascii_letters
    new_text = [x for x in text if x in allowed]
    return ''.join(new_text)


# fill out the function below that transforms the input text and window-size
# into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    p = len(text) - window_size
    inputs = [text[x:window_size+x] for x in range(0, p, step_size)]
    outputs = [text[window_size+x] for x in range(0, p, step_size)]

    return inputs, outputs


# build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars), name='LSTM_hobbit_1'))
    model.add(Dense(num_chars, activation='softmax', name='Dense_hobbit_1'))
    return model
