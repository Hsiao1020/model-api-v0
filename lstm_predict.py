import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from talib import abstract

from data import get_price_data, move_to_first_column, divide_into_train_and_test, split_sequence, dataframe_to_price_and_time, str_to_unixtime
from model import build_LSTM_Model
import json

ONE_DAY_MILLISECONDS = 86400000


def mean_absolute_percentage_error(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return f"{round(np.mean(np.abs((actual - pred) / actual)) * 100, 2)}%"


def predict(begin, end, features, OHLC, predict_ticket, index_features, ratio_of_train, look_back, forecast_days, layers, learning_rate, epochs, batch_size):
    data = get_price_data(begin, end, features)
    df = data[OHLC]

    if type(df) is pd.core.frame.DataFrame:
        df = move_to_first_column(df, predict_ticket)
    elif type(df) is pd.core.series.Series:
        df = df.to_frame(name=predict_ticket)

    if index_features:
        ohlc_df = get_price_data(begin, end, predict_ticket)
        ohlc_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                       'Close': 'close', 'Volume': 'volume'}, inplace=True)
        ohlc_df = ohlc_df.astype('float')
        for ta in index_features:
            output = eval('abstract.'+ta+'(ohlc_df)')
            # 如果輸出是一維資料，幫這個指標取名為 x 本身；多維資料則不需命名
            output.name = ta.lower() if type(output) == pd.core.series.Series else None
            df = pd.concat([df, output], axis=1)
            # except:
            #     print(f'error: {ta}')

    df = df.fillna(method='ffill')  # 假日時可能為 NA 值，使用前面最近的值取代
    df = df.fillna(method='bfill')  # 若還是有 NA 值，使用後面最近的值取代

    print(df)

    # 分訓練集、測試集
    df_train, df_test = divide_into_train_and_test(df, ratio_of_train)

    # 依據前測天數和預測天數切分資料
    X_train, Y_train = split_sequence(
        df_train.values, look_back, forecast_days)
    X_test, Y_test = split_sequence(df_test.values, look_back, forecast_days)

    # X 資料集正規化
    X_scaler = MinMaxScaler(feature_range=(0, 1))

    nsamples, nx, ny = X_train.shape
    d2_X_train = X_train.reshape((nsamples, nx*ny))
    X_train_norm = X_scaler.fit_transform(d2_X_train)
    X_train_norm = X_train_norm.reshape(nsamples, nx, ny)

    nsamples, nx, ny = X_test.shape
    d2_X_test = X_test.reshape((nsamples, nx*ny))
    X_test_norm = X_scaler.transform(d2_X_test)
    X_test_norm = X_test_norm.reshape(nsamples, nx, ny)

    forecast_data = np.array(df_test)[-look_back-1:-1]
    nx, ny = forecast_data.shape
    d2_forecast_data = forecast_data.reshape((1, nx*ny))
    forecast_data_norm = X_scaler.transform(d2_forecast_data)
    forecast_data_norm = forecast_data_norm.reshape(1, nx, ny)

    # Y 資料集正規化
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_train_norm = Y_scaler.fit_transform(Y_train)
    # Y_test_norm = Y_scaler.transform(Y_test)

    lstm_model = build_LSTM_Model(
        look_back, forecast_days, len(df.columns), layers, learning_rate)
    lstm_model.summary()

    history = lstm_model.fit(X_train_norm, Y_train_norm,
                             epochs=epochs, batch_size=batch_size)

    # 預測訓練集資料
    train_prediction = lstm_model.predict(X_train_norm)
    # 訓練集反正規化
    train_prediction = Y_scaler.inverse_transform(train_prediction)
    d_train = {
        'Actual ': Y_train.flatten(),
        'BTC Price Predictions': train_prediction.flatten()
    }
    train_result = pd.DataFrame(
        data=d_train, index=df.index[look_back:(look_back+len(train_prediction))])

    # 預測測試集資料
    test_prediction = lstm_model.predict(X_test_norm)
    # Y資料集反正規化
    test_prediction = Y_scaler.inverse_transform(test_prediction)
    d_test = {
        'Actual ': Y_test.flatten(),
        'BTC Price Predictions': test_prediction.flatten()
    }
    test_result = pd.DataFrame(
        data=d_test, index=df.index[-len(test_prediction):])

    # 預測未來資料
    forecast_prediction = lstm_model.predict(forecast_data_norm)
    # 未來資料反正規化
    forecast_prediction = float(
        Y_scaler.inverse_transform(forecast_prediction)[0])

    mse = mean_squared_error(test_prediction.flatten(), Y_test.flatten())
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(
        Y_test.flatten(), test_prediction.flatten())
    print(f"rmse: {rmse}")
    print(f"mape: {mape}")
    train_data_price, train_data_time = dataframe_to_price_and_time(
        train_result, 'BTC Price Predictions')
    test_data_price, test_data_time = dataframe_to_price_and_time(
        test_result, 'BTC Price Predictions')

    js = {
        'begin': str_to_unixtime(begin),
        'end': str_to_unixtime(end),
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'predict_data_train_price': train_data_price,
        'predict_data_train_time': train_data_time,
        'predict_data_test_price': test_data_price,
        'predict_data_test_time': test_data_time,
        'predict_data_forecast_price': forecast_prediction,
        'predict_data_forecast_time': test_data_time[-1]+ONE_DAY_MILLISECONDS,
        'history': history.history
    }

    return js
