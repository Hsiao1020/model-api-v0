import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import logging

def build_LSTM_Model(look_back, forecast_days,  n_features, layers, learning_rate=0.0001):
    model = Sequential()
    model.add(InputLayer(input_shape=(look_back, n_features)))
    for i in range(1, len(layers)+1):
        # units
        if i == len(layers):
            model.add(LSTM(units = layers[i-1]))
        else:
            model.add(LSTM(layers[i-1], return_sequences=True))
        # dropout
        # dropout = layers[i-1]['dropout']
        # if dropout and type(dropout) is float and 1 > dropout > 0:
        #     model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanSquaredError(), RootMeanSquaredError()])
    return model