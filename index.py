from data import get_price_data, dataframe_to_json

import talib
from talib import abstract
import pandas as pd
import numpy as np

def calculate_all_index(begin, end, feature):
    df = get_price_data(begin, end, [feature])
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    df = df.astype('float')
    ta_list = talib.get_functions()
    if 'MAVP' in ta_list:
        ta_list.remove('MAVP')
    for index in ta_list:
        try:
            output = eval('abstract.'+index+'(df)')
            # 如果輸出是一維資料，幫這個指標取名為 x 本身；多維資料則不需命名
            output.name = index.lower() if type(output) == pd.core.series.Series else None
            # 透過 merge 把輸出結果併入 df DataFrame
            df = pd.merge(df, pd.DataFrame(output), left_on=df.index, right_on=output.index)
            df = df.set_index('key_0')
        except:
            print(f'error: {index}')
    all_index_data = {}
    for index in df.columns:
        if index in ['open', 'high', 'low', 'close', 'Adj Close', 'volume']:
            pass
        else:
            all_index_data[index] = dataframe_to_json(df, index)

    return {
        "begin": begin,
        "end": end,
        "feature": feature,
        "all_index_data": all_index_data
    }


def calculate_index(begin, end, feature, index):
    ta_list = talib.get_functions()
    ta_list.remove('MAVP')
    if index not in ta_list:
        return {"error": "Can't find this index"}

    df = get_price_data(begin, end, [feature])
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df = df.astype('float')
    output = eval('abstract.'+index+'(df)')
    # 如果輸出是一維資料，幫這個指標取名為 x 本身；多維資料則不需命名
    output.name = index.lower() if type(output) == pd.core.series.Series else None
    df_output = pd.DataFrame(output)

    index_data = {}
    for i in df_output.columns:
        if i in ['open', 'high', 'low', 'close', 'Adj Close', 'volume']:
            pass
        else:
            index_data[i] = dataframe_to_json(df_output, i)
    return {
        "begin": begin,
        "end": end,
        "feature": feature,
        "index": index,
        "index": index_data
    }


def calculate_moving_average(begin, end, feature, timeperiod):
    df = get_price_data(begin, end, [feature])
    result = talib.MA(df.Close, timeperiod).to_frame()

    moving_average = dataframe_to_json(result, 0)

    return {
        "begin": begin,
        "end": end,
        "feature": feature,
        "index": "Moving Average",
        "timeperiod": timeperiod,
        "moving_average": moving_average
    }
