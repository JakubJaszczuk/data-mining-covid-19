import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformations import (
    drop_nonmonotonic_columns, deaggregate_values, shift_to_day_zero,
    smooth_values_ewm, smooth_values_rolling, count_anomalies,
    replace_anomalies, add_total, get_total, prepare_from_file,
    normalize_data
)


def plot(title, x, y, file=None):
    #plt.xticks(rotation=30)
    plt.ticklabel_format(style='plain', axis='y')
    plt.ticklabel_format(style='plain', axis='y')
    plt.suptitle(title)
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


def main():
    df: pd.DataFrame = pd.read_csv('./final_confirmed.csv', index_col='date', parse_dates=True)
    death: pd.DataFrame = pd.read_csv('./final_death.csv', index_col='date', parse_dates=True)
    cured: pd.DataFrame = pd.read_csv('./final_cured.csv', index_col='date', parse_dates=True)
    df, deaggregated = prepare_from_file(df)
    df_d, deaggregated_d = prepare_from_file(death)
    df_c, deaggregated_c = prepare_from_file(cured)

    # Select
    countries = ['Poland', 'Germany', 'China', 'Czechia']
    countries_most = ['Total', 'US', 'India', 'Brazil', 'France']
    join = df[countries]
    join_de = deaggregated[countries]
    join_m = df[countries_most]
    join_m_de = deaggregated[countries_most]

    # Merge Total
    total = pd.DataFrame({'Cases': df['Total'], 'Deaths': df_d['Total'], 'Recovered': df_c['Total']})
    deaggregated_total = pd.DataFrame({'Cases': deaggregated['Total'], 'Deaths': deaggregated_d['Total'], 'Recovered': deaggregated_c['Total']})

    # Plotting
    sns.set_theme()
    sns.lineplot(data=join)
    plot("Całkowita liczba przypadków dla wybranych krajów", "Data", "Przypadki", 'selected')

    sns.lineplot(data=join_de)
    plot("Dzienna liczba przypadków dla wybranych krajów", "Data", "Przypadki", 'selected_de')

    sns.lineplot(data=total)
    plot("Całkowita liczba zachorowań, zmarłych i wyzdrowiałych", "Data", "Przypadki", 'total')

    sns.lineplot(data=deaggregated_total)
    plot("Dzienna liczba zachorowań, zmarłych i wyzdrowiałych", "Data", "Przypadki", 'total_de')
    #d = deaggregated.loc[np.datetime64('2020-12-10')].drop('Total')


if __name__ == "__main__":
    main()
