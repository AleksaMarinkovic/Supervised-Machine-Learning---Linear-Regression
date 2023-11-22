import os
import pandas as pd
import numpy as np
from plots import plot_with_column
from normalization import zscore_normalize_features
from linearRegression import compute_cost, compute_gradient, gradient_descent
import matplotlib.pyplot as plt

# Data path
current_directory = os.getcwd()
relative_path = 'PolovniAutomobili.xlsx'
file_path = os.path.join(current_directory, relative_path)

# Columns to read from xlsx
cols_to_read = ['Year','Mileage','Fuel type','Displacement', 'Power', 'Registration', 'Damage', 'Price']

df = pd.read_excel(file_path, engine='openpyxl',usecols=cols_to_read)

# Drop rows with NaN values
df.dropna(inplace=True)

# Drop rows which don't contain price
df = df[df.iloc[:, -1] != 'Po dogovoru']

# Convert strings with characters and numbers to just numbers and cast as int
def convert_string_to_int(distance_str):
    return int(''.join(filter(str.isdigit, str(distance_str))))

# Convert year to int
df['Year'] = df['Year'].astype(int)

# Remove the kilometer part from for example '322.213 km' -> 322213 and convert to int
df['Mileage'] = df['Mileage'].apply(convert_string_to_int)

# Factorize fuel type. Takes unique values and assings a number from 1 - numberOfUniqueValues
df['Fuel type'] = pd.factorize(df['Fuel type'])[0]

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
df['Damage'] = pd.factorize(df['Damage'])[0]

# Remove euro sign and dot from the price and convert to int
df['Price'] = df['Price'].str.replace('€', '').str.replace('.', '').astype(int)

# Convert DataFrame to numpy array
numpy_array = df.to_numpy()

# Take last column as Y values
y_train = numpy_array[:,-1]

# Take every column but last column as X values
x_train = numpy_array[:,:-1]

# Normalize with standard deviation and mean and return the mu (mean vector) and sigma (standard deviation vector) for future use with predictions
x_train_normalized, mu, sigma = zscore_normalize_features(x_train)

# Get the prices column
prices_normalized_column = x_train_normalized[:,0]
# New engineered prices columns
prices_2_normalized_column = prices_normalized_column**2
prices_3_normalized_column = prices_normalized_column**3

# Get the mileage column
mileage_normalized_column = x_train_normalized[:,1]
# New engineered mileage columns
mileage_reciprocal_normalized_column = np.reciprocal(mileage_normalized_column)

# Get the mileage column
fuel_type_normalized_column = x_train_normalized[:,2]
# New engineered mileage columns
fuel_type_reciprocal_normalized_column = np.reciprocal(mileage_normalized_column)

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
x_train_eg_normalized = np.insert(x_train_normalized,x_train_normalized.shape[1],[prices_2_normalized_column,prices_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1],[displacement_2_normalized_column, displacement_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1],[power_2_normalized_column, power_3_normalized_column], axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1], mileage_reciprocal_normalized_column, axis=1)
x_train_eg_normalized = np.insert(x_train_eg_normalized,x_train_eg_normalized.shape[1], fuel_type_reciprocal_normalized_column, axis=1)
cols_= ['Year','Mileage','Fuel type','Displacement', 'Power', 'Registration', 'Damage', 'Price squared', 'Price qubed', 'Displacement squared', 'Displacement qubed','Power squared','Power qubed', 'Mileage reciprocal','Fuel type reciprocal']

print(x_train_eg_normalized.shape)

initial_w = np.zeros_like(x_train_eg_normalized[0,:])
initial_b = 0
iterations = 10000
alpha = 2.0e-4

w_final, b_final, J_hist = gradient_descent(x_train_eg_normalized, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)


m,_ = x_train_eg_normalized.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train_eg_normalized[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

#for i in range(0, x_train_normalized.shape[1]):
#    plt.scatter(x_train_normalized[:,i], y_train, marker='x', c='r', label="Actual Value"); plt.title(f"Price/{cols_[i]}")
#    plt.scatter(x_train_normalized[:,i], np.dot(x_train_eg_normalized[:,i], w_final[i]) + b_final, label="Predicted Value", marker='o',c='b'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
#Plot the correlation
#for i in range(0,x_train_normalized.shape[1]):
#    plot_with_column(x_train_normalized, y_train, i, df.columns[i], f"Price/{df.columns[i]}")