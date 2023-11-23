import os
import pandas as pd
import numpy as np
from plots import plot_with_column
from normalization import zscore_normalize_features
from linearRegression import compute_cost, compute_gradient, gradient_descent, compute_cost_linear_reg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
np.set_printoptions(threshold=np.inf)

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

current_year = datetime.now().year

# Convert year to int
df['Year'] = df['Year'].apply(convert_to_age)
# Get the year column
year_normalized_column = df['Year']
# New engineered year columns and
df['Year squared'] = year_normalized_column**2
df['Year cubed'] = year_normalized_column**3

# Remove the kilometer part from for example '322.213 km' -> 322213 and convert to int
df['Mileage'] = df['Mileage'].apply(convert_string_to_int)
# Get the mileage column
mileage_normalized_column = df['Mileage']
# Take only the kW value from for example '111/151 (kW/KS)' -> 111 and convert to int
df['Power'] = df['Power'].str.split('/').str[0].str.extract('(\d+)').astype(float).astype(int)
# Get the power column
power_normalized_column = df['Power']
# New engineered power columns
df['Power squared'] = power_normalized_column**2
df['Power cubed'] = power_normalized_column**3

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

# Transform left wheel to 2 and right wheel to 1 (not 1 0 so the weight can punish if right wheel)
def transform_leftwheel_to_bool(value):
    if pd.isna(value) or value == 'Levi volan':
        return 2
    else:
        return 1

columns = list(df.columns)
df['Wheel side'] = df['Wheel side'].apply(transform_leftwheel_to_bool)
# Move the specified column to the second last position
columns.remove('Wheel side')
columns.insert(-1, 'Wheel side')

# Converts date values to 1 and when 'Nije registrovan' is the value convert it to 0 
def transform_registration_to_bool(value):
    if pd.isna(value) or value == 'Nije registrovan':
        return 0
    else:
        return 1

df['Registration'] = df['Registration'].apply(transform_registration_to_bool)
columns.remove('Registration')
columns.insert(-1, 'Registration')

# Reorder the DataFrame columns
df = df[columns]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

y = df['Price']  # Target variable
df = df.drop('Price', axis=1) 

df = df.astype(int)

X = df  # Features

original_column_names = df.columns[:].tolist()

X = X.to_numpy()
y = y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize with standard deviation and mean and return the mu (mean vector) and sigma (standard deviation vector) for future use with predictions
X_train, mu, sigma = zscore_normalize_features(X_train)
X_test, mu, sigma = zscore_normalize_features(X_test)

# initial_w = np.array([ 7.75217913e+02, -1.94213188e+02,  2.45832408e+03,  3.16060242e+02,
#   1.12923792e+03, -6.71065203e+01,  4.10589895e+02, -2.42545471e+01,
#  -3.05294975e+02, -3.15474338e+00, -1.25378505e+02, -2.50166895e+01,
#  -9.70841495e+01,  5.39625307e+01, -1.42708247e+02,  1.36311596e+02,
#   1.54468846e+01,  1.49833533e+03, -4.02679855e+01, -6.88176632e+00,
#  -4.10800270e+02,  4.10480996e+00,  5.52871749e+02,  8.68559766e+01,
#  -2.99996760e+01,  3.75279413e+02, -5.01808748e+01,  1.75151448e+02,
#  -4.77967638e+01, -3.17082550e+02, -2.47830336e+02, -8.35836479e+01,
#  -6.16008219e+01, -2.54239649e+02, -2.48569193e+02, -1.96611460e+02,
#   1.39231694e+02,  2.31939066e+01, -1.33087203e+02,  1.01573296e+02,
#  -1.57515581e+02,  5.56696251e+02, -1.80723237e+02,  2.77943458e+02,
#  -3.44113190e+02, -1.51804098e+02, -8.27534250e+00,  2.58430536e+01,
#   1.08268700e+02, -1.56231234e+02, -4.75419987e+01, -5.36655438e+02,
#   1.10832456e+02,  1.71611602e+02,  4.02517378e+02,  3.86612540e+02,
#   7.02551486e+02, -3.82714044e+02, -3.04913412e+02,  8.74515412e+02,
#  -4.93719545e+01, -1.13611990e+02, -6.08519870e+02,  1.91097749e+01,
#  -1.91097749e+01, -6.55794106e+01,  1.75326747e+01,  7.92900252e+01,
#  -2.37996668e+01, -6.62229590e+00,  3.38336228e+01, -1.30114190e+02,
#  -9.25210661e+01,  1.87910726e+01,  1.23639693e+02, -2.31857855e+02,
#   3.40644424e+01,  2.07175891e+01, -8.13882458e+01, -1.57718253e+02,
#  -2.55069536e+02, -4.22811145e+01, -4.60976571e+01,  4.38584159e+01,
#   1.78367800e+00,  2.14630998e+02,  7.11804922e+01,  1.75424729e+02,
#   7.90124004e+00,  1.48095891e+01, -6.16737837e+01,  8.99226637e+01,
#   7.62000632e+02, -5.32890099e+01, -7.61208516e+01, -3.07390372e+01,
#   1.08269951e+02, -5.24978294e+02,  2.55178917e+02,  2.17171879e+02,
#  -1.38372951e+02, -1.66374535e+02,  2.79443339e+03,  8.67637198e+02,
#   1.06276300e+03, -1.15883194e+02,  4.56748160e+00])
initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 10000
alpha = 1.0e-4
lambda_ = 1.0e0
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg, compute_gradient, alpha, iterations, lambda_)

for i in range(X_train.shape[0]):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}, year: {X_train[i,0]}, mileage: {X_train[i,1]}, power: {X_train[i,2]}")


#for i in range(0, x_train_normalized.shape[1]):
#    plt.scatter(x_train_normalized[:,i], y_train, marker='x', c='r', label="Actual Value"); plt.title(f"Price/{cols_[i]}")
#    plt.scatter(x_train_normalized[:,i], np.dot(x_train_eg_normalized[:,i], w_final[i]) + b_final, label="Predicted Value", marker='o',c='b'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print(f"b,w found by gradient descent: {b_final:0.2f},{np.array2string(w_final, separator=', ')}")
for i in range(len(original_column_names)):
    print(f"Weight for '{original_column_names[i]}': {w_final[i]:0.2f}")

#Plot the correlation
#for i in range(0,x_train_normalized.shape[1]):
#    plot_with_column(x_train_normalized, y_train, i, df.columns[i], f"Price/{df.columns[i]}")