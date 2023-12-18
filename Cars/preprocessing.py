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
price_filter = 100000
displacement_filter = 8000
def get_data():
    # Data path
    current_directory = os.getcwd()
    relative_path = 'PolovniAutomobili.xlsx'
    file_path = os.path.join(current_directory, relative_path)

    # Columns to read from xlsx
    cols_to_read = ['Manufacturer', 'Year', 'Mileage', 'Chasis','Fuel type', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration', 'Damage', 'Price']

    df = pd.read_excel(file_path, engine='openpyxl', usecols=cols_to_read)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    # Drop rows which don't contain price
    df = df[df.iloc[:, -1] != 'Po dogovoru']
    df = df[df.iloc[:, -1] != 'Na upit']
    # Drop rows containing electric cars
    df = df[df['Fuel type'] != 'Električni pogon']

    df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(float)
    df = df[df['Price'] <= price_filter]
    df['Price'] = np.log(df['Price'])

    # Convert strings with characters and numbers to just numbers and cast as float
    def convert_string_to_float(distance_str):
        return float(''.join(filter(str.isdigit, str(distance_str))))

    # Convert model year to age
    def convert_to_age(year_made):
        year = float(datetime.now().year)-float(year_made)
        if year == 0: year = 1
        return year
    
    # Convert year to float
    df['Year'] = df['Year'].apply(convert_to_age)    
    df = df[df['Year'] <= year_filter]

    # Remove the kilometer part from for example '322.213 km' -> 322213 and convert to float
    df['Mileage'] = df['Mileage'].apply(convert_string_to_float)
    df = df[df['Mileage'] <= mileage_filter]
    #df['Mileage_Squared'] = df['Mileage']**2
    #df['Mileage_Cubed'] = df['Mileage']**3
    
    # Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to float
    df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(float)
    df = df[df['Power'] <= power_filter]


    # df['Displacement'] = df['Displacement'].str.replace(' cm3', '').astype(float)
    # 
    # # Drop rows whose displacement is unreal
    # df = df[df['Displacement'] < displacement_filter]
    
    # Convert to 'Nije registrovan' if null or 'Nije registrovan', otherwise set to 'Registrovan'
    def transform_registration_to_bool(value):
        if pd.isna(value) or value == 'Nije registrovan':
            return 'Nije registrovan'
        else:
            return 'Registrovan'

    df['Registration'] = df['Registration'].apply(transform_registration_to_bool)
    # plotBarGraphs(df,5,3, ['Manufacturer', 'Chasis' ,'Fuel type', 'Drivetrain', 'Transmission', 'Emissions','Wheel side', 'Color', 'Registration', 'Damage'])
    plotHistograms(df, 1,4,["Year","Mileage","Power","Price"],["Year","KM","KW","Euros"],height=800, width=1400)
    # Onehot encode Registration
    df = pd.get_dummies(df, columns=['Registration'], prefix='Registration')

    # Onehot encode Wheel side
    df = pd.get_dummies(df, columns=['Wheel side'], prefix='Wheel side')

    # Onehot encode Manufacturer
    df = pd.get_dummies(df, columns=['Manufacturer'], prefix='Manufacturer')

    #Onehot encode Model
    # df = pd.get_dummies(df, columns=['Model'], prefix='Model')

    # Onehot encode Chasis
    df = pd.get_dummies(df, columns=['Chasis'], prefix='Chasis')

    # Onehot encode Emissions
    df = pd.get_dummies(df, columns=['Emissions'], prefix='Emissions')

    # Onehot encode Drivetrain
    df = pd.get_dummies(df, columns=['Drivetrain'], prefix='Drivetrain')

    # Onehot encode Transmissions
    df = pd.get_dummies(df, columns=['Transmission'], prefix='Transmission')

    # Onehot encode Number of doors
    #df = pd.get_dummies(df, columns=['Number of doors'], prefix='Number of doors')

    # Onehot encode Number of seats
    #df = pd.get_dummies(df, columns=['Number of seats'], prefix='Number of seats')

    # Onehot encode Air conditioning
    #df = pd.get_dummies(df, columns=['Air conditioning'], prefix='Air conditioning')

    # Onehot encode Color
    df = pd.get_dummies(df, columns=['Color'], prefix='Color')

    # Onehot encode Fuel type
    df = pd.get_dummies(df, columns=['Fuel type'], prefix='Fuel')

    # Onehot encode Damage
    df = pd.get_dummies(df, columns=['Damage'], prefix='Damage')   

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
   
    df = df.astype(float)

    return df

# TODO easily fit data to be predicted

def fit_data_to_dataset(emtpy_dataframe, X):
    emtpy_dataframe['Year'] = X['Year']
    emtpy_dataframe['Mileage'] = X['Mileage']
    emtpy_dataframe['Power'] = X['Power']
    for i in range(X.shape[0]):
        emtpy_dataframe.loc[i,[f'Registration_{X.loc[i, "Registration"]}']] = 1
        emtpy_dataframe.loc[i,[f'Wheel side_{X.loc[i, "Wheel side"]}']] = 1
        emtpy_dataframe.loc[i,[f'Manufacturer_{X.loc[i, "Manufacturer"]}']] = 1
        emtpy_dataframe.loc[i,[f'Chasis_{X.loc[i, "Chasis"]}']] = 1
        emtpy_dataframe.loc[i,[f'Emissions_{X.loc[i, "Emissions"]}']] = 1
        emtpy_dataframe.loc[i,[f'Drivetrain_{X.loc[i, "Drivetrain"]}']] = 1
        emtpy_dataframe.loc[i,[f'Transmission_{X.loc[i, "Transmission"]}']] = 1
        emtpy_dataframe.loc[i,[f'Color_{X.loc[i, "Color"]}']] = 1
        emtpy_dataframe.loc[i,[f'Fuel_{X.loc[i, "Fuel"]}']] = 1
        emtpy_dataframe.loc[i,[f'Damage_{X.loc[i, "Damage"]}']] = 1
    print(emtpy_dataframe.head())
    X = emtpy_dataframe.to_numpy()
    return X