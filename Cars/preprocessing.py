import os
import numpy as np
import pandas as pd
from datetime import datetime
from plots import plotHistograms, plotBarGraphs

mileage_filter = 600000
power_filter = 300
year_filter = 25
price_filter = 40000
def get_train_and_test_data():
    # Data path
    current_directory = os.getcwd()
    relative_path = 'PolovniAutomobili.xlsx'
    file_path = os.path.join(current_directory, relative_path)

    # Columns to read from xlsx
    cols_to_read = ['Manufacturer', 'Model', 'Year', 'Displacement','Mileage', 'Chasis' ,'Fuel type', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Number of doors', 'Number of seats', 'Wheel side','Air conditioning', 'Color', 'Registration', 'Damage', 'Price']

    df = pd.read_excel(file_path, engine='openpyxl',usecols=cols_to_read)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    # Drop rows which don't contain price
    df = df[df.iloc[:, -1] != 'Po dogovoru']
    df = df[df.iloc[:, -1] != 'Na upit']

    df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(int)
    df = df[df['Price'] <= price_filter]

    # Convert strings with characters and numbers to just numbers and cast as int
    def convert_string_to_int(distance_str):
        return int(''.join(filter(str.isdigit, str(distance_str))))

    # Convert model year to age
    def convert_to_age(year_made):
        return int(datetime.now().year)-int(year_made)+1
    
    # Convert year to int
    df['Year'] = df['Year'].apply(convert_to_age)    
    df = df[df['Year'] <= year_filter]
    df['Year_Sqaured'] = df['Year']**2
    df['Year_Cubed'] = df['Year']**3

    df['Displacement'] = df['Displacement'].str.replace(' cm3', '').astype(int)
    df['Displacement_Sqaured'] = df['Displacement']**2
    df['Displacement_Cubed'] = df['Displacement']**3

    # Remove the kilometer part from for example '322.213 km' -> 322213 and convert to int
    df['Mileage'] = df['Mileage'].apply(convert_string_to_int)
    df = df[df['Mileage'] <= mileage_filter]
    df['Mileage_Sqaured'] = df['Mileage']**2
    df['Mileage_Cubed'] = df['Mileage']**3
    
    # Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to int
    df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(int)
    df = df[df['Power'] <= power_filter]
    df['Power_Sqaured'] = df['Power']**2
    df['Power_Cubed'] = df['Power']**3

    df['MileagePerYear'] = df['Mileage'] / df['Year']
    df['MileagePerYear_Sqaured'] = df['MileagePerYear']**2
    df['MileagePerYear_Cubed'] = df['MileagePerYear']**3
    
    # Convert to 'Nije registrovan' if null or 'Nije registrovan', otherwise set to 'Registrovan'
    def transform_registration_to_bool(value):
        if pd.isna(value) or value == 'Nije registrovan':
            return 'Nije registrovan'
        else:
            return 'Registrovan'

    df['Registration'] = df['Registration'].apply(transform_registration_to_bool)

    plotBarGraphs(df,5,3, ['Manufacturer', 'Model', 'Chasis' ,'Fuel type', 'Emissions', 'Drivetrain', 'Transmission', 'Number of doors', 'Number of seats', 'Wheel side','Air conditioning', 'Color', 'Registration', 'Damage'])
    plotHistograms(df, 1,5,["Year","Mileage","Power","Price","Displacement"],["Year","KM","KW","Euros","cm3"],height=800, width=1400)

    # Onehot encode Registration
    df = pd.get_dummies(df, columns=['Registration'], prefix='Registration')

    # Onehot encode Wheel side
    df = pd.get_dummies(df, columns=['Wheel side'], prefix='Wheel side')

    # Onehot encode Manufacturer
    df = pd.get_dummies(df, columns=['Manufacturer'], prefix='Manufacturer')

    # Onehot encode Model
    df = pd.get_dummies(df, columns=['Model'], prefix='Model')

    # Onehot encode Chasis
    df = pd.get_dummies(df, columns=['Chasis'], prefix='Chasis')

    # Onehot encode Emissions
    df = pd.get_dummies(df, columns=['Emissions'], prefix='Emissions')

    # Onehot encode Drivetrain
    df = pd.get_dummies(df, columns=['Drivetrain'], prefix='Drivetrain')

    # Onehot encode Transmissions
    df = pd.get_dummies(df, columns=['Transmission'], prefix='Transmission')

    # Onehot encode Number of doors
    df = pd.get_dummies(df, columns=['Number of doors'], prefix='Number of doors')

    # Onehot encode Number of seats
    df = pd.get_dummies(df, columns=['Number of seats'], prefix='Number of seats')

    # Onehot encode Air conditioning
    df = pd.get_dummies(df, columns=['Air conditioning'], prefix='Air conditioning')

    # Onehot encode Color
    df = pd.get_dummies(df, columns=['Color'], prefix='Color')

    # Onehot encode Fuel type
    df = pd.get_dummies(df, columns=['Fuel type'], prefix='Fuel')

    # Onehot encode Damage
    df = pd.get_dummies(df, columns=['Damage'], prefix='Damage')   

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
   
    df = df.astype(int)

    return df