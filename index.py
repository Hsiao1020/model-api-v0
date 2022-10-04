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
    df = get_price_data(begin, end, [feature])
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df = df.astype('float')

    ta_list = talib.get_functions()
    ta_list.remove('MAVP')

    for i in index:
        if i not in ta_list:
            return {"error": f"Can't find index {i}"}
        # try:
        output = eval('abstract.'+i+'(df)')
        # 如果輸出是一維資料，幫這個指標取名為 x 本身；多維資料則不需命名
        output.name = i.lower() if type(output) == pd.core.series.Series else None
        # 透過 merge 把輸出結果併入 df DataFrame
        # df_output = pd.concat([df_output, output], axis=1)
        df = pd.concat([df, output], axis=1)
        # except:
        #     print(f'error: {index}')

    index_data = {}
    for i in df.columns:
        if i in ['open', 'high', 'low', 'close', 'Adj Close', 'volume']:
            pass
        else:
            index_data[i] = dataframe_to_json(df, i)

    return {
        "begin": begin,
        "end": end,
        "feature": feature,
        "index": index,
        "index_data": index_data
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
