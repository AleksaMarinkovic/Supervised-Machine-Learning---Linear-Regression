import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from normalization import zscore_normalize_features
import seaborn as sns

mileage_filter = 1500000

def get_train_and_test_data():
    # Data path
    current_directory = os.getcwd()
    relative_path = 'PolovniAutomobili.xlsx'
    file_path = os.path.join(current_directory, relative_path)

    # Columns to read from xlsx
    cols_to_read = ['Manufacturer', 'Year','Mileage', 'Chasis' ,'Fuel type', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Number of doors', 'Number of seats', 'Wheel side','Air conditioning', 'Color', 'Registration', 'Damage', 'Price']

    df = pd.read_excel(file_path, engine='openpyxl',usecols=cols_to_read)


    # Drop rows with NaN values
    df.dropna(inplace=True)
    # Drop rows which don't contain price
    df = df[df.iloc[:, -1] != 'Po dogovoru']
    df = df[df.iloc[:, -1] != 'Na upit']

    df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(int)

    # Convert strings with characters and numbers to just numbers and cast as int
    def convert_string_to_int(distance_str):
        return int(''.join(filter(str.isdigit, str(distance_str))))


    def convert_to_age(year_made):
        return int(datetime.now().year)-int(year_made)+1

    # Convert year to int
    df['Year'] = df['Year'].apply(convert_to_age)
    # Remove the kilometer part from for example '322.213 km' -> 322213 and convert to int
    df['Mileage'] = df['Mileage'].apply(convert_string_to_int)
    df = df[df['Mileage'] <= mileage_filter]
    # Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to int
    df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(int)

    # Converts date values to 1 and when 'Nije registrovan' is the value convert it to 0 
    def transform_registration_to_bool(value):
        if pd.isna(value) or value == 'Nije registrovan':
            return 0
        else:
            return 1

    df['Registration'] = df['Registration'].apply(transform_registration_to_bool)

    X_original = df


    y = df['Price']  # Target variable
    df = df.drop('Price', axis=1) 
    # Onehot encode Wheel side
    df = pd.get_dummies(df, columns=['Wheel side'], prefix='Wheel side')

    # Onehot encode Manufacturer
    df = pd.get_dummies(df, columns=['Manufacturer'], prefix='Manufacturer')

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

    X = df  # Features

    original_column_names = cols_to_read
    preprocessed_column_names = df.columns[:].tolist()

    column_names_for_heatmap = ['Year','Mileage','Power']
    df_for_heatmap = df.filter(regex='^Transmission', axis = 1)
    df_for_heatmap = pd.concat([df_for_heatmap, df[column_names_for_heatmap]],axis=1)
    #showHeatMap(df_for_heatmap)
    X = X.to_numpy()
    y_original = pd.DataFrame(y, columns=['Price'])
    y = y.to_numpy()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Normalize with standard deviation and mean and return the mu (mean vector) and sigma (standard deviation vector) for future use with predictions
    X_train, mu_train, sigma_train = zscore_normalize_features(X_train, 3)
    X_test, mu_test, sigma_test = zscore_normalize_features(X_test, 3)

    # Add engineered columns to train data
    X_train_year_column = X_train[:,0]
    X_train_year_squared = X_train_year_column **2
    X_train_year_cubed = X_train_year_column **3

    X_train_power_column = X_train[:,2]
    X_train_power_squared = X_train_power_column **2
    X_train_power_cubed = X_train_power_column **3

    X_train = np.column_stack((X_train, X_train_year_squared))
    X_train = np.column_stack((X_train, X_train_year_cubed))

    X_train = np.column_stack((X_train, X_train_power_squared))
    X_train = np.column_stack((X_train, X_train_power_cubed))

    # Add engineered columns to test data
    X_test_year_column = X_test[:,0]
    X_test_year_squared = X_test_year_column **2
    X_test_year_cubed = X_test_year_column **3

    X_test_power_column = X_test[:,2]
    X_test_power_squared = X_test_power_column **2
    X_test_power_cubed = X_test_power_column **3

    X_test = np.column_stack((X_test, X_test_year_squared))
    X_test = np.column_stack((X_test, X_test_year_cubed))

    X_test = np.column_stack((X_test, X_test_power_squared))
    X_test = np.column_stack((X_test, X_test_power_cubed))

    # Add column names
    original_column_names.append("Year squared")
    original_column_names.append("Year cubed")
    original_column_names.append("Power squared")
    original_column_names.append("Power cubed")

    return X_train, X_test, y_train, y_test, mu_train, sigma_train, mu_test, sigma_test, X_original, y_original, original_column_names, preprocessed_column_names

def plotHistograms(dataframe, rows, cols, column_names, numerical_data_names, title="Distribution of numerical data", height=1400, width=800):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            fig.add_trace(go.Histogram(x=dataframe[str(column_names[i*cols+j])], name=numerical_data_names[i*cols+j]), row=i+1, col=j+1) if j + i*cols < len(column_names) else None

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()

def plotBarGraphs(dataframe, rows, cols, column_names, title="Distribution of categorical data", height=1400, width=1400):
    print(dataframe.info())
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=column_names)
    for i in range(0, rows):
        for j in range(0, cols):
            if j + i * cols < len(column_names):
                count = dataframe[str(column_names[i*cols+j])].value_counts().reset_index()
                fig.add_trace(go.Bar(y=count['count'], x=count[str(column_names[i*cols+j])]), row=i+1, col=j+1)

    fig.update_layout(height=height, width=width, title_text=title)
    fig.show()

def showHeatMap(dataframe):
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
    plt.show()