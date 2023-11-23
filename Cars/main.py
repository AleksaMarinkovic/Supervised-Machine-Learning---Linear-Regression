import os
import pandas as pd
import numpy as np
from plots import plot_with_column
from normalization import zscore_normalize_features
from linearRegression import compute_cost, compute_gradient, gradient_descent, compute_cost_linear_reg
import matplotlib.pyplot as plt

# Data path
current_directory = os.getcwd()
relative_path = 'PolovniAutomobili.xlsx'
file_path = os.path.join(current_directory, relative_path)

# Columns to read from xlsx
cols_to_read = ['Manufacturer', 'Year','Mileage', 'Chasis' ,'Fuel type','Displacement', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Number of doors', 'Number of seats', 'Air conditioning', 'Color', 'Registration', 'Damage', 'Price']

df = pd.read_excel(file_path, engine='openpyxl',usecols=cols_to_read)

# Drop rows with NaN values
df.dropna(inplace=True)
# Drop rows which don't contain price
df = df[df.iloc[:, -1] != 'Po dogovoru']
df = df[df.iloc[:, -1] != 'Na upit']

price_column = df.pop('Price')
# Convert strings with characters and numbers to just numbers and cast as int
def convert_string_to_int(distance_str):
    return int(''.join(filter(str.isdigit, str(distance_str))))


# Onehot encode Manufacturer
df = pd.get_dummies(df, columns=['Manufacturer'], prefix='Manufacturer')

# Convert year to int
df['Year'] = df['Year'].astype(int)

# Remove the kilometer part from for example '322.213 km' -> 322213 and convert to int
df['Mileage'] = df['Mileage'].apply(convert_string_to_int)

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

# Remove the cm3 part from for example '1998 cm3' -> 1998 and convert to int
df['Displacement'] = df['Displacement'].str.extract('(\d+)').astype(float).astype(int)

# Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to int
df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(int)

# Converts date values to 1 and when 'Nije registrovan' is the value convert it to 0 
def transform_to_bool(value):
    if pd.isna(value) or value == 'Nije registrovan':
        return 0
    else:
        return 1

# Apply the above function to registration
df['Registration'] = df['Registration'].apply(transform_to_bool)

# Factorize damage type. 
#df['Damage'] = pd.factorize(df['Damage'])[0]
df = pd.get_dummies(df, columns=['Damage'], prefix='Damage')

# Remove euro sign and dot from the price and convert to int
df['Price'] = price_column
df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(int)

df = df.astype(int)
# Convert DataFrame to numpy array
numpy_array = df.to_numpy()

# Take last column as Y values
y_train = numpy_array[:,-1]

# Take every column but last column as X values
x_train = numpy_array[:,:-1]
# Normalize with standard deviation and mean and return the mu (mean vector) and sigma (standard deviation vector) for future use with predictions
x_train_normalized, mu, sigma = zscore_normalize_features(x_train)

# Get the year column
year_normalized_column = x_train_normalized[:,0]
# New engineered prices columns
year_2_normalized_column = year_normalized_column**2
year_3_normalized_column = year_normalized_column**3

# Get the mileage column
mileage_normalized_column = x_train_normalized[:,1]
# New engineered mileage columns
mileage_reciprocal_normalized_column = np.reciprocal(mileage_normalized_column)

# Get the displacement column
displacement_normalized_column = x_train_normalized[:,3]
# New engineered displacement columns
displacement_2_normalized_column = displacement_normalized_column**2
displacement_3_normalized_column = displacement_normalized_column**3

# Get the power column
power_normalized_column = x_train_normalized[:,4]
# New engineered power columns
power_2_normalized_column = power_normalized_column**2
power_3_normalized_column = power_normalized_column**3

# Add the engineered columns
x_train_eg_normalized = np.insert(x_train_normalized,x_train_normalized.shape[1],[year_2_normalized_column,year_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1],[displacement_2_normalized_column, displacement_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1],[power_2_normalized_column, power_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1], mileage_reciprocal_normalized_column, axis=1)
cols_= ['Year','Mileage','Fuel type','Displacement', 'Power', 'Registration', 'Damage', 'Price squared', 'Price qubed', 'Displacement squared', 'Displacement qubed','Power squared','Power qubed', 'Mileage reciprocal','Fuel type reciprocal']

# Normalize the engineered features
initial_w = np.array([ 2.46396223e+03, -7.07643731e+01,  2.96371252e+01,  7.85755799e+02,
  1.23653941e+03,  1.93577233e+00,  4.39819488e+02, -3.81750214e+02,
 -1.05052396e+02,  2.49093932e+00, -1.43240442e+02,  4.51791340e+01,
 -7.98401789e+01, -2.99731815e+01,  2.83722589e+02, -1.15463929e+02,
 -1.05869764e+02, -1.40218687e+02,  4.22515221e+00, -5.74494181e+02,
  2.94201735e+01, -2.49904324e+02, -7.82426449e+01,  1.44872966e+02,
  4.80296350e+02,  1.72565286e+01, -9.09032552e+01,  7.51217389e+01,
 -2.27773317e+02, -2.32833253e+02,  1.34389245e+01, -8.77926469e+01,
 -1.08098720e+02, -4.46740641e+02, -2.36626978e+02,  1.09981802e+02,
  4.24204567e+02, -2.26541462e+02, -2.10754723e-01,  1.37436760e+02,
  1.28385717e+03, -5.39900166e+02,  1.33925598e+02, -2.56781249e+02,
 -2.23613532e+02, -2.33531540e+02,  8.84004892e+00,  1.51785016e+02,
 -3.51887159e+02, -1.73535407e+02, -3.17735099e+02, -1.60913252e+02,
 -2.01432268e+02,  9.89413782e+02,  1.39026744e+02,  5.90536574e+02,
 -2.74526834e+02, -9.05494085e+01,  6.37566663e+02, -9.97107846e+01,
 -1.24821443e+02, -3.92287131e+02,  2.15742239e+02, -2.15742239e+02,
 -1.54041449e+02, -6.21976494e+00, -4.61345708e+01,  1.14291191e+02,
 -6.82505125e+00, -3.38405974e+01, -4.44198235e+01,  5.35397347e+01,
 -1.26641582e+01, -4.02557421e+02,  2.85669871e+02,  7.46038107e+01,
 -1.79114202e+02, -1.49889854e+02, -1.55528985e+02, -2.52259171e+02,
 -9.87983339e+01, -3.62878618e+00,  8.46854972e+01,  2.82079750e+02,
  3.46652524e+02,  1.65332147e+02,  1.58472486e+02,  1.19039838e+02,
  9.13763960e+01, -9.32757789e+01,  1.07356673e+01, -1.98739777e+02,
  1.48246020e+02, -6.95391748e+01,  1.17257319e+02,  2.64903843e+02,
 -8.27939787e+01, -2.61203014e+02,  1.44303180e+03,  2.14306572e+02,
  9.70913147e+02,  1.55509182e+01,  1.19629437e+03, -7.32743260e+01,
  5.67089717e+00])
#initial_w = np.zeros(x_train_eg_normalized.shape[1])
initial_b = 2550.17
iterations = 10000
alpha = 2.0e-4

w_final, b_final, J_hist = gradient_descent(x_train_eg_normalized, y_train, initial_w, initial_b, compute_cost_linear_reg, compute_gradient, alpha, iterations, 2)

for i in range(x_train_eg_normalized.shape[0]):
    print(f"prediction: {np.dot(x_train_eg_normalized[i], w_final) + b_final:0.2f}, target value: {y_train[i]}, year: {x_train[i,0]}, mileage: {x_train[i,1]}, displacement: {x_train[i,2]}")


#for i in range(0, x_train_normalized.shape[1]):
#    plt.scatter(x_train_normalized[:,i], y_train, marker='x', c='r', label="Actual Value"); plt.title(f"Price/{cols_[i]}")
#    plt.scatter(x_train_normalized[:,i], np.dot(x_train_eg_normalized[:,i], w_final[i]) + b_final, label="Predicted Value", marker='o',c='b'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print(f"b,w found by gradient descent: {b_final:0.2f},{np.array2string(w_final, separator=', ')}")
#Plot the correlation
#for i in range(0,x_train_normalized.shape[1]):
#    plot_with_column(x_train_normalized, y_train, i, df.columns[i], f"Price/{df.columns[i]}")