import numpy as np
from normalization import reverse_zscore_normalize_features
from linear_regression import gradient_descent, compute_cost_linear_reg, compute_gradient_reg, predict
from preprocessing import get_data, fit_data_to_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from normalization import zscore_normalize_features, normalize_new_data_point
from plots import plot_with_column, showHeatMap, showResult, plotResidualQQ, plotResidualHistogram, \
    plotResidualsAgainstFitted
import pandas as pd
import random
import time

np.set_printoptions(threshold=np.inf)

columns_to_normalize = 9

df, target_means_manufacturer, target_means_model = get_data()

cols_to_plot_with_column = ["Mileage", "Power", "PowerRoot", "Year", "MileagePerYear"]
cols_to_plot_heatmap = ["Price", "Mileage", "Power", "Year"]
# showHeatMap(df, cols_to_plot_heatmap)

for i in range(len(cols_to_plot_with_column)):
    plot_with_column(df, cols_to_plot_with_column[i])

y = df['Price']  # Target variable
df = df.drop('Price', axis=1)

df_for_predictions = pd.DataFrame(columns=df.columns)
row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df_for_predictions = pd.concat([df_for_predictions, row_with_zeros], ignore_index=True)

columns_for_new_predictions = ['Manufacturer', 'Model', 'PowerRoot', 'Year', 'Mileage', 'Displacement', 'PowerPerLitre',
                               'Chasis', 'Fuel', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Wheel side',
                               'Color', 'Registration']

new_row = {'Manufacturer': 'Kia', 'Year': 11, 'Model': 'Sportage', 'Mileage': 140000, 'Displacement': 1685,
           'Chasis': 'Džip/SUV', 'Fuel': 'Dizel', 'Power': 86,
           'Emissions': 'Euro 5', 'Drivetrain': 'Prednji', 'Transmission': 'Manuelni 6 brzina',
           'Wheel side': 'Levi volan', 'Color': 'Siva', 'Registration': 'Registrovan'}

X_new_for_prediction = pd.DataFrame([new_row], columns=columns_for_new_predictions)

X_processed_new_for_prediction = fit_data_to_dataset(df_for_predictions, X_new_for_prediction,
                                                     target_means_manufacturer, target_means_model)
X_processed_new_for_prediction = X_processed_new_for_prediction.squeeze()

X = df

X = X.to_numpy()
y = y.to_numpy()

preprocessed_column_names = df.columns[:].tolist()
print(" \nPRVIH 10 TAČAKA PRE NORMALIZACIJE (STANDARDIZACIJA): \n")
for i in range(10):
    row = X[i, :columns_to_normalize]
    formatted_row = [f"{val:.4f}" for val in row]
    print(" ".join(formatted_row))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

X_train, mu_train, sigma_train = zscore_normalize_features(X_train, columns_to_normalize)
X_test, mu_test, sigma_test = zscore_normalize_features(X_test, columns_to_normalize)

print(" \nPRVIH 10 TAČAKA POSLE NORMALIZACIJE (STANDARDIZACIJA): \n")
for i in range(10):
    row = X[i, :columns_to_normalize]
    formatted_row = [f"{val:.4f}" for val in row]
    print(" ".join(formatted_row))

initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 10000
alpha = 1.0e-2
lambda_ = 2e0

start_time = time.time()
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg,
                                            compute_gradient_reg, alpha, iterations, lambda_)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Process took: {elapsed_time} seconds")

# For displaying purposes
X_train_reversed = reverse_zscore_normalize_features(X_train, mu_train, sigma_train, columns_to_normalize)
X_test_reversed = reverse_zscore_normalize_features(X_test, mu_test, sigma_test, columns_to_normalize)

train_percentage = []
y_predicted_train = np.array([])
for i in range(X_train_reversed.shape[0]):
    row = np.exp(np.dot(X_train[i], w_final) + b_final)
    y_train_reversed = np.exp(y_train[i])

    individual_train_percentage = (y_train_reversed * 100) / row
    train_percentage.append(individual_train_percentage)
    y_predicted_train = np.append(y_predicted_train, np.array(row))

test_percentage = []

y_predicted_test = np.array([])

for i in range(X_test.shape[0]):
    row = np.exp(np.dot(X_test[i], w_final) + b_final)
    y_test_reversed = np.exp(y_test[i])

    individual_test_percentage = (y_test_reversed * 100) / row
    test_percentage.append(individual_test_percentage)
    y_predicted_test = np.append(y_predicted_test, np.array(row))
    if 80 <= individual_test_percentage <= 120:
        prediction_string_color = '\033[92m'
    elif 60 <= individual_test_percentage <= 80 or 120 <= individual_test_percentage <= 140:
        prediction_string_color = '\033[93m'
    else:
        prediction_string_color = '\033[31m'
    prediction_string = f'{prediction_string_color}{individual_test_percentage}{prediction_string_color}'
    print(
        f"\033[94mtest {i} prediction: {row:0.2f}, target value: {y_test_reversed:0.2f}, year: {X_test_reversed[i, 0]}"
        f", mileage: {X_test_reversed[i, 1]}, power: {X_test_reversed[i, 3]}\033[94m, "
        f"prediction percentage: {prediction_string}")

residuals = y_predicted_test - np.exp(y_test)
plotResidualQQ(residuals)
plotResidualHistogram(residuals)

plotResidualsAgainstFitted(residuals, y_predicted_test)

print(f"\033[97mb,w found by gradient descent: {b_final:0.2f},{np.array2string(w_final, separator=', ')}\033[97m")

for i in range(len(preprocessed_column_names)):
    print(f"Weight for '{preprocessed_column_names[i]}': {w_final[i]:0.2f}")

y_predicted_train = y_predicted_train.reshape(-1, 1)
y_predicted_test = y_predicted_test.reshape(-1, 1)
print("\n")
print("\033[93mTRAIN\033[93m \033[92mMEAN PERCENTAGE: \033[92m")
print(np.mean(train_percentage))
print("\033[94mTEST\033[94m \033[92mMEAN PERCENTAGE: \033[92m")
print(np.mean(test_percentage))
print("\033[97m")

print(f"\033[96mR2_SCORE={r2_score(np.exp(y_test), y_predicted_test)}\033[96m")

print("\n")
print("\033[97m")
print(
    f"PREDICTION FOR NEW VALUE: {np.exp(predict(normalize_new_data_point(X_processed_new_for_prediction, mu_train, sigma_train, columns_to_normalize), w_final, b_final)):0.2f}")
