import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import tensorflow as tf
from transformations import (
    smooth_values_ewm, smooth_values_rolling, prepare_from_file, normalize_data
)
from model import Dataset


def predictions_to_dataframe(predictions, begin_date):
    dates = pd.date_range(start=begin_date, periods=len(predictions), freq='D')
    return pd.DataFrame({'predictions': predictions}, index=dates)


def plot(dataset, title, x, y, file=None, input_dates=None, predictions=None, begin=None):
    #plt.xticks(rotation=30)
    sns.lineplot(data=dataset.train_df.rename(columns=lambda x: 'train'))
    sns.lineplot(data=dataset.valid_df.rename(columns=lambda x: 'valid'), palette=['red'])
    sns.lineplot(data=dataset.test_df.rename(columns=lambda x: 'test'), palette=['seagreen'])
    if input_dates is not None:
        sns.lineplot(data=input_dates, palette=['purple'], linewidth=3)
    if predictions is not None:
        sns.scatterplot(data=predictions, palette=['teal'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.tight_layout()
    #if file is None:
    if True:
        plt.show()
    else:
        plt.savefig(f'{file}.png')
    plt.close()


def plot_prediction(dataset, title, x, y, file=None, input_dates=None, predictions=None, begin=None):
    sns.lineplot(data=dataset.df.rename(columns=lambda x: 'Dane'))
    if input_dates is not None:
        sns.lineplot(data=input_dates.rename(columns=lambda x: 'Wejście modelu'), palette=['purple'], linewidth=3)
    if predictions is not None:
        sns.scatterplot(data=predictions.rename(columns=lambda x: 'Predykcja'), palette=['teal'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(title)
    plt.axvline(pd.to_datetime('2020-09-16'), color='red', label='Granica zb. uczącego')
    plt.axvline(pd.to_datetime('2020-12-10'), color='seagreen', label='Granica zb. walidacyjnego')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    #if file is None:
    if True:
        plt.show()
    else:
        plt.savefig(f'{file}.png')
    plt.close()


def load_model(filename):
    path = f'./models/{filename}'
    return tf.keras.models.load_model(path)


def to_model_inputs(raw, date_range):
    inputs_df = raw.loc[date_range].rename(columns=lambda x: 'input_range')
    inputs = inputs_df.to_numpy()
    inputs = np.expand_dims(inputs, 0)
    return inputs_df, inputs


def main():
    sns.set_theme()
    df: pd.DataFrame = pd.read_csv('./final_confirmed.csv', index_col='date', parse_dates=True)
    df, deaggregated = prepare_from_file(df, np.float32)
    df = normalize_data(df)
    deaggregated = normalize_data(deaggregated)
    deaggregated_smooth = smooth_values_rolling(deaggregated)

    country = 'Poland'
    data = df[[country]]
    data_de = deaggregated[[country]]
    data_de_smooth = deaggregated_smooth[[country]]

    # Create data
    in_days = 60
    dataset = Dataset(data, in_days, 14, batch_size=32)
    dataset_de = Dataset(data_de, in_days, 14, batch_size=32)
    dataset_de_smooth = Dataset(data_de_smooth, in_days, 14, batch_size=32)
    model = load_model('agregat')
    model_de = load_model('diff')
    model_de_smooth = load_model('smooth')

    # Prepare inputs
    date_range_1 = pd.date_range(start='2020-05-01', periods=in_days, freq='D')
    date_range_2 = pd.date_range(start='2020-08-01', periods=in_days, freq='D')
    date_range_3 = pd.date_range(start='2020-10-01', periods=in_days, freq='D')
    date_range_4 = pd.date_range(start='2020-11-15', periods=in_days, freq='D')
    date_ranges = [date_range_1, date_range_2, date_range_3, date_range_4]

    inputs_cum = [to_model_inputs(data, dr) for dr in date_ranges]
    inputs_de = [to_model_inputs(data_de, dr) for dr in date_ranges]
    inputs_de_smooth = [to_model_inputs(data_de_smooth, dr) for dr in date_ranges]

    # Lista list par
    inputs = [inputs_cum, inputs_de, inputs_de_smooth]

    matplotlib.rcParams["savefig.directory"] = '/home/riper/Pulpit/9 semestr/Eksploracja danych/python/plots_final/'

    # Run prediction
    for i in inputs[0:1]:
        for dr, (ins_df, ins) in zip(date_ranges, i):
            res = model.predict(ins)
            res = res.squeeze()
            res = predictions_to_dataframe(res, dr[-1] + pd.Timedelta(days=1))
            plot_prediction(dataset, "Predykcja na tle prawdziwych danych", "Data", "Przypadki (skalowane)", input_dates=ins_df, predictions=res)

    for i in inputs[1:2]:
        for dr, (ins_df, ins) in zip(date_ranges, i):
            res = model_de.predict(ins)
            res = res.squeeze()
            res = predictions_to_dataframe(res, dr[-1] + pd.Timedelta(days=1))
            plot_prediction(dataset_de, "Predykcja na tle prawdziwych danych", "Data", "Przypadki (skalowane)", input_dates=ins_df, predictions=res)

    for i in inputs[2:3]:
        for dr, (ins_df, ins) in zip(date_ranges, i):
            res = model_de_smooth.predict(ins)
            res = res.squeeze()
            res = predictions_to_dataframe(res, dr[-1] + pd.Timedelta(days=1))
            plot_prediction(dataset_de_smooth, "Predykcja na tle prawdziwych danych", "Data", "Przypadki (skalowane)", input_dates=ins_df, predictions=res)


if __name__ == "__main__":
    main()
