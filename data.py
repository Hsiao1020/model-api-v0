import numpy as np
import yfinance as yf
import pandas as pd


def get_price_data(begin, end, features):
    data = yf.download(features, start=begin, end=end, interval='1d')
    data.index = pd.DatetimeIndex(data.index).tz_localize('UTC').tz_convert('Asia/Taipei')
    # data.index = data.index.tz_localize('UTC').tz_convert('Asia/Taipei')
    data = data.fillna(method='ffill') # 假日時可能為 NA 值，使用前面最近的值取代
    data = data.fillna(method='bfill') # 若還是有 NA 值，使用後面最近的值取代
    return data


# 將要預測的欄位移至第一欄
def move_to_first_column(df, column_name):
    col = df.pop(column_name)
    df.insert(loc=0 , column=column_name, value=col)
    return df


def divide_into_train_and_test(df, ratio_of_train=0.7):
    train_size = int(ratio_of_train*len(df))
    train, test = df[0:train_size], df[train_size:len(df)]
    return train, test


# 根據輸入的天數切分資料
def split_sequence(data, look_back, forecast_days):
    X,Y = [],[]
    for i in range(0,len(data)-look_back-forecast_days +1):
        X.append(data[i:(i+look_back)])
        Y.append(data[i+look_back][0])
    return np.array(X), np.array(Y).reshape((np.array(Y).shape[0], 1))

def dataframe_to_json(df, column_name):
    result = []
    dt_index = df.index.view(np.int64) // 10**6
    for i in range(len(df)):
        if not np.isnan(df[column_name][i]):
            result.append([int(dt_index[i]), float(df[column_name][i])])
    return result
