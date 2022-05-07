import pandas as pd


def normaliseDataframe(df):
    arr = df.values
    dic = {'min' : arr.min(axis=0),
           'max' : arr.max(axis=0)
           }
    arr = (arr - dic['min']) / (dic['max'] - dic['min'])

    return pd.DataFrame(arr, columns=df.columns).set_index(df.index), dic


def denormaliseDataframe(df, scaler):
    arr = df.values
    arr = arr * (scaler['max'] - scaler['min']) + scaler['min']
    return pd.DataFrame(arr, columns=df.columns).set_index(df.index)