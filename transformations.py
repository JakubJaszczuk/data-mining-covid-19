import pandas as pd
import numpy as np


def count_anomalies(df: pd.DataFrame):
    x = df > df.shift(-1)
    return {'unique': x.any().sum(), 'all': x.sum().sum()}


def check_monotonic(df: pd.DataFrame):
    return ~(df > df.shift(-1)).any()


def all_monotonic(df: pd.DataFrame):
    return check_monotonic(df).all()


def replace_anomalies(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    diff = df > df.shift(-1)  # Obecny jest większy niż następny
    indices = np.where(diff)
    for row, col in zip(*indices):
        # Poprzedni jak istnieje to czy mniejszy
        if row-1 >= 0 and df.iloc[row-1, col] <= df.iloc[row+1, col]:
            # Interpoluj
            avg = (df.iloc[row-1, col] + df.iloc[row+1, col]) // 2
            df.iloc[row, col] = avg
    return df


def drop_nonmonotonic_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = (df > df.shift(-1)).any()
    return df[df.columns[~x]]


def deaggregate_values(df: pd.DataFrame):
    return df.diff().fillna(0)


def smooth_values_ewm(df: pd.DataFrame, period: int = 7):
    # Exponential weighted window
    return df.ewm(span=7).mean()


def smooth_values_resample(df: pd.DataFrame, period: str = '7D'):
    return df.resample(period).mean()  # median


def smooth_values_rolling(df: pd.DataFrame, period: str = '7D'):
    return df.rolling(period).mean()  # Można inne funkcje okna


def shift_to_day_zero(df: pd.DataFrame):
    s = df.ne(0).cumsum().eq(0).sum()
    return df.apply(lambda x: x.shift(periods=-s[x.name], fill_value=0)).reset_index(drop=True)


def get_total(df: pd.DataFrame):
    return df.sum(axis=1)


def add_total(df: pd.DataFrame):
    df['Total'] = df.sum(axis=1)


def with_total(df: pd.DataFrame):
    return df.assign(total=df.sum(axis=1))


def prepare_from_file(data, value_type=np.int64):
    df = replace_anomalies(data)
    df = df.astype(value_type)
    add_total(df)
    deaggregated = deaggregate_values(df)
    return df, deaggregated


def normalize_data(df: pd.DataFrame, kind='minmax'):
    if kind == 'minmax':
        return (df - df.min()) / (df.max() - df.min())
    elif kind == 'standard':
        return (df - df.mean()) / df.std()
    else:
        raise ValueError


def load_prepared():
    df: pd.DataFrame = pd.read_csv('./final_confirmed.csv', index_col='date', parse_dates=True)
    df.index = df.index.to_period('D')
    df = df.astype(np.int64)
    d1 = drop_nonmonotonic_columns(df)
    d1 = deaggregate_values(d1)
    mono1 = all_monotonic(df)
    mono2 = all_monotonic(d1)
    print(mono1, mono2)
    print(df)
    print(d1)
    #pl = df['Poland']
    #res = df.apply(lambda c: np.trim_zeros(c, trim='f'), axis='columns', result_type='expand')
    #res = [np.trim_zeros(column[1], trim='f') for column in df.iteritems()]
    #a = pd.DataFrame(res)
    dfs = shift_to_day_zero(df)
    print(dfs)


def main():
    load_prepared()


if __name__ == "__main__":
    main()
