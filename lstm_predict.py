import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from data import get_price_data, move_to_first_column, divide_into_train_and_test, split_sequence
from model import build_LSTM_Model

def predict(begin, end, features, OHLC, predict_ticket, ratio_of_train, look_back, forecast_days, layers, learning_rate, epochs, batch_size):
    data = get_price_data(begin, end, features)
    df = data[OHLC]

    if type(df) is pd.core.frame.DataFrame:
        df = move_to_first_column(df, predict_ticket)
    elif type(df) is pd.core.series.Series:
        df = df.to_frame(name=predict_ticket)

    # 分訓練集、測試集
    df_train, df_test = divide_into_train_and_test(df, ratio_of_train)
    # 依據前測天數和預測天數切分資料
    X_train, Y_train = split_sequence(df_train.values, look_back, forecast_days)
    X_test, Y_test = split_sequence(df_test.values, look_back, forecast_days)
    # X 資料集正規化
    X_scaler = MinMaxScaler(feature_range=(0, 1))

    nsamples, nx, ny = X_train.shape
    d2_X_train = X_train.reshape((nsamples,nx*ny))
    X_train_norm = X_scaler.fit_transform(d2_X_train)
    X_train_norm = X_train_norm.reshape(nsamples,nx,ny)

    nsamples, nx, ny = X_test.shape
    d2_X_test = X_test.reshape((nsamples,nx*ny))
    X_test_norm = X_scaler.transform(d2_X_test)
    X_test_norm = X_test_norm.reshape(nsamples,nx,ny)


    # Y 資料集正規化
    Y_scaler= MinMaxScaler(feature_range=(0, 1))
    Y_train_norm = Y_scaler.fit_transform(Y_train)
    Y_test_norm = Y_scaler.transform(Y_test)

    lstm_model = build_LSTM_Model(look_back, forecast_days, len(features), layers, learning_rate)
    lstm_model.summary()

    history = lstm_model.fit(X_train_norm, Y_train_norm, epochs=epochs, batch_size=batch_size)

    # 預測訓練集資料
    train_prediction = lstm_model.predict(X_train_norm)
    # X資料集反正規化
    X_pred_scaler = MinMaxScaler()
    X_pred_scaler.min_, X_pred_scaler.scale_ = X_scaler.min_[0], X_scaler.scale_[0]
    train_prediction = X_pred_scaler.inverse_transform(train_prediction)
    d_train = {'Actual ': Y_train.flatten(), 'BTC Price Predictions': train_prediction.flatten()}
    train_result = pd.DataFrame(data=d_train, index=df_train.index[-len(train_prediction):])

    # 預測測試集資料
    test_prediction = lstm_model.predict(X_test_norm)
    # Y資料集反正規化
    test_prediction = Y_scaler.inverse_transform(test_prediction)
    d_test = {'Actual ': Y_test.flatten(), 'BTC Price Predictions': test_prediction.flatten()}
    test_result = pd.DataFrame(data=d_test, index=df_test.index[-len(test_prediction):])

    mse = mean_squared_error(test_prediction.flatten(), Y_test.flatten())
    rmse = np.sqrt(mse)
    print(f"mse: {mse}")
    print(f"rmse: {rmse}")
    js = {
    'mse': mse,
    'rmse': rmse,
    'predict_data_train': json.loads(train_result['BTC Price Predictions'].to_json(orient='index')),
    'predict_data_test': json.loads(test_result['BTC Price Predictions'].to_json(orient='index'))}
    return js
