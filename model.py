import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import logging

def build_LSTM_Model(look_back, forecast_days,  n_features, layers, learning_rate=0.0001):
   # get TF logger
    # Add the output handler.
    # logger = tf.get_logger()
    # logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler('./tensorflow.log')
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)
    # logger.debug("????????????????????")
    model = Sequential()
    model.add(InputLayer(input_shape=(look_back, n_features)))
    for i in range(1, len(layers)+1):
        # units
        if i == len(layers):
            model.add(LSTM(units = layers[i-1]['units']))
        else:
            model.add(LSTM(layers[i-1]['units'], return_sequences=True))
        # dropout
        dropout = layers[i-1]['dropout']
        if dropout and type(dropout) is float and 1 > dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[RootMeanSquaredError()])
    return model