import os
import numpy as np
import pandas as pd
from datetime import datetime
from plots import plotHistograms, plotBarGraphs
import matplotlib.pyplot as plt
from normalization import zscore_normalize_features

mileage_filter = 600000
power_filter = 300
year_filter = 30
price_filter = 40000
displacement_filter = 4000


def get_data():
    # Data path
    current_directory = os.getcwd()
    relative_path = 'Vozila.xlsx'
    file_path = os.path.join(current_directory, relative_path)

    # Columns to read from xlsx
    cols_to_read = ['Year', 'Mileage', 'Displacement', 'Manufacturer', 'State', 'Model', 'Chasis', 'Fuel type', 'Power',
                    'Emissions',
                    'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration', 'Damage', 'Price']

    df = pd.read_excel(file_path, engine='openpyxl', usecols=cols_to_read)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    # Drop rows containing electric cars, damaged cars and cars without a listed price
    df = df[df['Price'] != 'Po dogovoru']
    df = df[df['State'] == 'Polovno vozilo'].drop(columns=['State'])
    df = df[df['Price'] != 'Na upit']
    df = df[df['Fuel type'] != 'Električni pogon']
    df = df[df['Damage'] == 'Nije oštećen'].drop(columns=['Damage'])

    df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(float)
    df = df[df['Price'] <= price_filter]

    df['Price'] = np.log(df['Price'])

    # Convert strings with characters and numbers to just numbers and cast as float
    def convert_string_to_float(distance_str):
        return float(''.join(filter(str.isdigit, str(distance_str))))

    # Convert model year to age
    def convert_to_age(year_made):
        year = float(datetime.now().year) - float(year_made)
        if year == 0: year = 1
        return year

    # Convert year to float
    df['Year'] = df['Year'].apply(convert_to_age)
    df = df[df['Year'] <= year_filter]

    # Remove the kilometer part from for example '322.213 km' -> 322213 and convert to float
    df['Mileage'] = df['Mileage'].apply(convert_string_to_float)
    df = df[df['Mileage'] <= mileage_filter]

    # Mileage per year
    df['MileagePerYear'] = df['Mileage'] / df['Year']

    # Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to float
    df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(float)
    df = df[df['Power'] <= power_filter]

    df['PowerRoot'] = np.sqrt(df['Power'])

    df['Displacement'] = df['Displacement'].str.replace(' cm3', '').astype(float)
    # 
    # Drop rows whose displacement is unreal
    df = df[df['Displacement'] < displacement_filter]

    df['PowerPerLitre'] = df['Power'] / (df['Displacement'] / 1000)

    # Convert to 'Nije registrovan' if null or 'Nije registrovan', otherwise set to 'Registrovan'
    def transform_registration_to_bool(value):
        if pd.isna(value) or value == 'Nije registrovan':
            return 'Nije registrovan'
        else:
            return 'Registrovan'

    df['Registration'] = df['Registration'].apply(transform_registration_to_bool)
    # plotBarGraphs(df,5,3, ['Manufacturer', 'Model', 'Chasis' ,'Fuel type', 'Drivetrain', 'Transmission', 'Emissions', 'Wheel side', 'Color', 'Registration'])
    # plotHistograms(df, 1, 3, ["Year", "Mileage", "Power"], ["Year", "KM", "KW"], height=800,
    #               width=1400)
    # plotHistograms(df, 1, 1, ["Price"], ["Euros"], height=800,
    #               width=500)

    target_means_manufacturer = df.groupby('Manufacturer')['Price'].mean().to_dict()
    # Replace the 'Manufacturer' column with the target-encoded values
    df['Manufacturer'] = df['Manufacturer'].map(target_means_manufacturer)

    target_means_model = df.groupby('Model')['Price'].mean().to_dict()
    # Replace the 'Model' column with the target-encoded values
    df['Model'] = df['Model'].map(target_means_model)

    # Onehot encode rest of categorical data
    df = pd.get_dummies(df, columns=['Registration', 'Wheel side', 'Chasis', 'Emissions', 'Drivetrain', 'Transmission',
                                     'Color', 'Fuel type'], prefix=['Registration', 'Wheel side', 'Chasis', 'Emissions',
                                                                    'Drivetrain', 'Transmission', 'Color', 'Fuel'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df = df.astype(float)

    column_names = df.columns.tolist()
    print(column_names)
    return df, target_means_manufacturer, target_means_model


def fit_data_to_dataset(new_dataframe, X, target_means_manufacturer, target_means_model):
    new_dataframe['Year'] = X['Year']
    new_dataframe['Mileage'] = X['Mileage']
    new_dataframe['MileagePerYear'] = X['Mileage'] / X['Year']
    new_dataframe['Power'] = X['Power']
    new_dataframe['Displacement'] = X['Displacement']
    new_dataframe['PowerRoot'] = np.sqrt(X['Power'])
    new_dataframe['PowerPerLitre'] = X['Power'] / (X['Displacement'] / 1000)
    new_dataframe['Model'] = X['Model'].map(target_means_model)
    new_dataframe['Manufacturer'] = X['Manufacturer'].map(target_means_manufacturer)

    for i in range(X.shape[0]):
        new_dataframe.loc[i, [f'Wheel side_{X.loc[i, "Wheel side"]}']] = 1
        new_dataframe.loc[i, [f'Chasis_{X.loc[i, "Chasis"]}']] = 1
        new_dataframe.loc[i, [f'Emissions_{X.loc[i, "Emissions"]}']] = 1
        new_dataframe.loc[i, [f'Drivetrain_{X.loc[i, "Drivetrain"]}']] = 1
        new_dataframe.loc[i, [f'Transmission_{X.loc[i, "Transmission"]}']] = 1
        new_dataframe.loc[i, [f'Color_{X.loc[i, "Color"]}']] = 1
        new_dataframe.loc[i, [f'Fuel_{X.loc[i, "Fuel"]}']] = 1
    X = new_dataframe.to_numpy()
    return X
