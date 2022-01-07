import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformations import (
    drop_nonmonotonic_columns, deaggregate_values, shift_to_day_zero,
    smooth_values_ewm, smooth_values_rolling, count_anomalies,
    replace_anomalies, add_total, prepare_from_file, normalize_data
)
from plots import plot


def split_data_into_groups(df: pd.DataFrame, train, val):
    n = len(df)
    s1 = int(n * train)
    s2 = int(n * val)
    train = df[0:s1]
    val = df[s1:s2]
    test = df[s2:]
    num_features = df.shape[1] if len(df.shape) == 2 else 1
    return train, val, test, num_features


def make_dataset(df: pd.DataFrame, in_size: int, out_size: int, batch_size: int, shuffle: bool=False, ):
    data = df.to_numpy()
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=in_size + out_size,
        shuffle=shuffle,
        batch_size=batch_size,
        sampling_rate=1,
    )
    ds = ds.map(lambda f: split_window(f, in_size, out_size))
    return ds


def split_window(features, in_size, out_size):
    inputs = features[:, slice(0, in_size), :]
    labels = features[:, slice(in_size, None), :]
    inputs.set_shape([None, in_size, None])
    labels.set_shape([None, out_size, None])
    return inputs, labels


class Dataset:
    def __init__(self, df: pd.DataFrame, in_size, out_size, split_1=0.7, split_2=0.95, batch_size=32):
        in_size = len(df.index) if in_size is None else in_size
        train, valid, test, features_count = split_data_into_groups(df, split_1, split_2)
        self.train = make_dataset(train, in_size, out_size, batch_size=batch_size)
        self.valid = make_dataset(valid, in_size, out_size, batch_size=batch_size)
        self.test = make_dataset(test, in_size, out_size, batch_size=batch_size)
        self.df = df
        self.train_df = train
        self.valid_df = valid
        self.test_df = test

        self.in_size = in_size
        self.out_size = out_size
        self.shift = out_size
        self.predict_days = out_size
        self.size = in_size + out_size
        self.batch_size = batch_size
        self.features = features_count


class Model:
    def __init__(self, data: Dataset, layer_units=32, iters=256, name='model'):
        self.data = data
        self.predict_days = data.predict_days
        self.features = data.features
        self.layer_units = layer_units
        self.iters = iters
        self.name = name
        self.model, self.history = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.layer_units, return_sequences=False),
            #tf.keras.layers.LSTM(self.layer_units, return_sequences=True),
            #tf.keras.layers.LSTM(self.layer_units * 2, return_sequences=False),
            tf.keras.layers.Dense(self.predict_days * self.features, kernel_initializer=tf.initializers.zeros(), activation=tf.nn.swish),
            tf.keras.layers.Reshape([self.predict_days, self.features]),
        ])
        model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.mse, metrics=[tf.metrics.mae])
        #saver = tf.keras.callbacks.ModelCheckpoint(filepath=f'./models/{self.name}', monitor='val_mean_absolute_error', save_best_only=True)
        saver = tf.keras.callbacks.ModelCheckpoint(filepath=f'/tmp/models_tf/{self.name}', monitor='val_mean_absolute_error', save_best_only=True)
        history = model.fit(self.data.train, validation_data=self.data.valid, epochs=self.iters, callbacks=[saver])
        #history = model.fit(self.data.train, validation_data=self.data.valid.repeat(), epochs=self.iters, validation_steps=2)
        return model, history

    def save_model(self, name):
        self.model.save(f"./models/{name}")

    def plot_history(self):
        sns.lineplot
        plt.show()


def main():
    sns.set_theme()
    df: pd.DataFrame = pd.read_csv('./final_confirmed.csv', index_col='date', parse_dates=True)
    df, deaggregated = prepare_from_file(df, np.float32)
    df = normalize_data(df)
    deaggregated = normalize_data(deaggregated)

    # Uśrednianie, wygładzanie
    deaggregated = smooth_values_rolling(deaggregated)

    # Wziąć Polskę i Niemcy
    country = 'Poland'
    data = df[[country]]
    data_de = deaggregated[[country]]

    # Create data
    epochs = 4096
    layers = 60
    dataset = Dataset(data_de, 60, 14, batch_size=32)
    model_de = Model(dataset, layers, epochs, name='smooth')
    history = model_de.history.history
    mae = np.array(history['mean_absolute_error'])
    val_mae = np.array(history['val_mean_absolute_error'])
    sns.lineplot(x=range(epochs), y=mae)
    if 'val_mean_absolute_error' in history:
        sns.lineplot(x=range(epochs), y=val_mae)
    plt.tight_layout()
    plt.title("Proces uczenia - dane dzienne (uśrednione)")
    #plt.title("Proces uczenia - dane dzienne")
    plt.xlabel("Epoki")
    plt.ylabel("Średni błąd absolutny")
    plt.show()


if __name__ == "__main__":
    main()

