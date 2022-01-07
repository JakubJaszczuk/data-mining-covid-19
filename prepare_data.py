import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


path = Path('./COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/')


def get_file_list() -> list:
    files = list(path.glob('*.csv'))
    return sorted(files, key=filename_to_date)


def normalize_columns(df: pd.DataFrame):
    normalize_drop_unused(df)
    normalize_column_names(df)
    normalize_column_types(df)
    remove_bad_rows(df)


def normalize_drop_unused(df: pd.DataFrame):
    df.drop('FIPS', errors='ignore', inplace=True, axis=1)
    df.drop('Admin2', errors='ignore', inplace=True, axis=1)
    df.drop('Last_Update', errors='ignore', inplace=True, axis=1)
    df.drop('Lat', errors='ignore', inplace=True, axis=1)
    df.drop('Long_', errors='ignore', inplace=True, axis=1)
    df.drop('Combined_Key', errors='ignore', inplace=True, axis=1)
    df.drop('Incident_Rate', errors='ignore', inplace=True, axis=1)
    df.drop('Case_Fatality_Ratio', errors='ignore', inplace=True, axis=1)
    df.drop('Active', errors='ignore', inplace=True, axis=1)  # Opcjonalnie
    df.drop('Incidence_Rate', errors='ignore', inplace=True, axis=1)
    df.drop('Case-Fatality_Ratio', errors='ignore', inplace=True, axis=1)
    df.drop('Latitude', errors='ignore', inplace=True, axis=1)
    df.drop('Longitude', errors='ignore', inplace=True, axis=1)
    df.drop('Last Update', errors='ignore', inplace=True, axis=1)


def normalize_column_names(df: pd.DataFrame):
    cols = {
        'Confirmed': 'potwierdzone',
        'Deaths': 'zgony',
        'Recovered': 'wyzdrowiały',
        'Province_State': 'region',
        'Province/State': 'region',
        'Country_Region': 'kraj',
        'Country/Region': 'kraj',
    }
    df.rename(columns=cols, errors='ignore', inplace=True)


def normalize_column_types(df: pd.DataFrame):
    df['region'] = df['region'].astype('string')
    df['kraj'] = df['kraj'].astype('string')


def remove_bad_rows(df: pd.DataFrame):
    df.drop(df.loc[df['region'] == 'Recovered'].index, inplace=True)


def filename_to_date(path: Path):
    filename = path.name
    d = filename.split('.')[0].split('-')
    return datetime(month=int(d[0]), day=int(d[1]), year=int(d[2]))


def countries_with_provinces(df: pd.DataFrame):
    '''
    'US', 'Canada', 'United Kingdom', 'China',
    'Netherlands', 'Australia', 'Denmark', 'France'
    '''
    print(df[df['region'].isna() is False].kraj.unique())


def aggregate_countries_state_data(df: pd.DataFrame):
    '''
    Australia, China, Canada, US
    '''
    return df.groupby('kraj').sum()


def separate_timeseries(df: pd.DataFrame, date):
    confirmed = pd.DataFrame({'values': df['potwierdzone'], 'date': date})
    death = pd.DataFrame({'values': df['zgony'], 'date': date})
    cured = pd.DataFrame({'values': df['wyzdrowiały'], 'date': date})
    confirmed = confirmed.reset_index().pivot(index='date', values='values', columns='kraj')
    death = death.reset_index().pivot(index='date', values='values', columns='kraj')
    cured = cured.reset_index().pivot(index='date', values='values', columns='kraj')
    return (confirmed, death, cured)


def process_file(file: Path):
    date = pd.to_datetime(filename_to_date(file))
    df = pd.read_csv(file)
    normalize_columns(df)
    agg = aggregate_countries_state_data(df)
    return separate_timeseries(agg, date)


def process_file_and_log(file: Path, i: int, total: int):
    print(f"Progress: {i+1}/{total}")
    return process_file(file)


def zeros_to_nans(df: pd.DataFrame):
    df.replace(0, np.nan, inplace=True)


def merge_duplicated_countries(df: pd.DataFrame):
    # Merge and drop
    countries = [
        ('Bahamas', 'Bahamas, The'),
        ('Bahamas', 'The Bahamas'),
        ('Gambia', 'Gambia, The'),
        ('Gambia', 'The Gambia'),
        ('Russia', 'Russian Federation'),
        ('South Korea', 'Republic of Korea'),
        ('South Korea', 'Korea, South'),
        ('Moldova', 'Republic of Moldova'),
        ('Iran', 'Iran (Islamic Republic of)'),
        ('Czechia', 'Czech Republic'),
        ('United Kingdom', 'UK'),
        ('China', 'Mainland China'),
        ('Taiwan', 'Taiwan*'),
        ('Taiwan', 'Taipei and environs'),
        ('Ivory Coast', "Cote d'Ivoire"),
    ]
    for c in countries:
        df[c[0]].fillna(df[c[1]], inplace=True)
        df.drop(columns=c[1], inplace=True)


def filter_uncomplete_data(df: pd.DataFrame):
    threshold = len(df.index) / 2
    df.dropna(axis='columns', thresh=threshold, inplace=True)


def clean_merged_data(df: pd.DataFrame):
    zeros_to_nans(df)
    merge_duplicated_countries(df)
    filter_uncomplete_data(df)
    df.fillna(0, inplace=True)
    #add_total(df)
    return df


def load_and_store():
    files: list = get_file_list()
    length = len(files)
    data = [process_file_and_log(file, i, length) for i, file in enumerate(files)]
    tables = map(list, zip(*data))
    confirmed, death, cured = [clean_merged_data(pd.concat(t)) for t in tables]
    confirmed.to_csv('./final_confirmed.csv')
    death.to_csv('./final_death.csv')
    cured.to_csv('./final_cured.csv')


def main():
    load_and_store()


if __name__ == "__main__":
    main()
