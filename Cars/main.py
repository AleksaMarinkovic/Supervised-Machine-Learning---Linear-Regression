import numpy as np
from normalization import reverse_zscore_normalize_features
from linear_regression import gradient_descent, compute_cost_linear_reg, compute_gradient_reg, predict
from preprocessing import get_data, fit_data_to_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from normalization import zscore_normalize_features, normalize_new_data_point
from plots import plot_with_column, showHeatMap, showResult, plotResidualQQ, plotResidualHistogram
import pandas as pd
import random
import time
np.set_printoptions(threshold=np.inf)

columns_to_normalize = 5

df = get_data()

cols_to_plot_with_column = ["Mileage", "Power", "PowerRoot", "Year"]
cols_to_plot_heatmap = ["Price", "Mileage", "Power", "Year"]
#showHeatMap(df, cols_to_plot_heatmap)

for i in range(len(cols_to_plot_with_column)):
    plot_with_column(df, cols_to_plot_with_column[i])

y = df['Price']  # Target variable
df = df.drop('Price', axis=1)

df_for_predictions = pd.DataFrame(columns=df.columns)
row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df_for_predictions = pd.concat([df_for_predictions, row_with_zeros], ignore_index=True)

columns_for_new_predictions = ['Manufacturer', 'Model', 'PowerRoot', 'Year', 'Mileage', 'Displacement', 'Chasis', 'Fuel', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration',
                               'Damage']

new_row = {'Manufacturer': 'Honda', 'Year': 7, 'Model': 'CR-V','Mileage': 50000, 'Displacement': 1998,'Chasis': 'Džip/SUV', 'Fuel': 'Dizel', 'Power': 118,
           'Emissions': 'Euro 6', 'Drivetrain': '4x4', 'Transmission': 'Automatski / poluautomatski',
           'Wheel side': 'Levi volan', 'Color': 'Bela', 'Registration': 'Registrovan', 'Damage': 'Nije oštećen'}

X_new_for_prediction = pd.DataFrame([new_row], columns=columns_for_new_predictions)


X_processed_new_for_prediction = fit_data_to_dataset(df_for_predictions, X_new_for_prediction)
X_processed_new_for_prediction = X_processed_new_for_prediction.squeeze()

X = df
X = X.to_numpy()
y = y.to_numpy()

preprocessed_column_names = df.columns[:].tolist()
X, mu, sigma = zscore_normalize_features(X, columns_to_normalize)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 10000
alpha = 5.0e-2
lambda_ = 0e0
start_time = time.time()
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg,
                                            compute_gradient_reg, alpha, iterations, lambda_)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Gradient descent took: {elapsed_time} seconds")
# For displaying purposes
X_train_reversed = reverse_zscore_normalize_features(X_train, mu, sigma, columns_to_normalize)
X_test_reversed = reverse_zscore_normalize_features(X_test, mu, sigma, columns_to_normalize)

train_percentage = []
y_predicted_train = np.array([])
for i in range(X_train_reversed.shape[0]):
    row = np.exp(np.dot(X_train[i], w_final) + b_final)
    y_train_reversed = np.exp(y_train[i])


    # row = np.dot(X_train[i], w_final) + b_final
    # y_train_reversed = y_train[i]

    # print(f"\033[93mtrain {i} prediction: {row:0.2f}, target value: {y_train[i]}, year: {X_train_reversed[i,0]}, mileage: {X_train_reversed[i,1]}, power: {X_train_reversed[i,2]}\033[93m")
    individual_train_percentage = (y_train_reversed * 100) / row if y_train_reversed < row else (
                                                                                                        row * 100) / y_train_reversed
    train_percentage.append(individual_train_percentage)
    y_predicted_train = np.append(y_predicted_train, np.array(row))

test_percentage = []
y_predicted_test = np.array([])
for i in range(X_test_reversed.shape[0]):
    row = np.exp(np.dot(X_test[i], w_final) + b_final)
    y_test_reversed = np.exp(y_test[i])

    # row = np.dot(X_test[i], w_final) + b_final
    # y_test_reversed = y_test[i]

    individual_test_percentage = (y_test_reversed * 100) / row if y_test_reversed < row else (
                                                                                                     row * 100) / y_test_reversed
    test_percentage.append(individual_test_percentage)
    y_predicted_test = np.append(y_predicted_test, np.array(row))
    if individual_test_percentage > 80:
        prediction_string_color = '\033[92m'
    elif individual_test_percentage > 60:
        prediction_string_color = '\033[93m'
    else:
        prediction_string_color = '\033[31m'
    prediction_string = f'{prediction_string_color}{individual_test_percentage}{prediction_string_color}'
    print(
        f"\033[94mtest {i} prediction: {row:0.2f}, target value: {y_test_reversed:0.2f}, year: {X_test_reversed[i, 0]}, mileage: {X_test_reversed[i, 1]}, power: {X_test_reversed[i, 3]}\033[94m, prediction percentage: {prediction_string}")


residuals = np.exp(y_test) - y_predicted_test
# residuals = y_test - y_predicted_test
plotResidualQQ(residuals)
plotResidualHistogram(residuals)

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
# print(f"\033[96mR2_SCORE={r2_score(y_test, y_predicted_test)}\033[96m")

print("\n")
print("\033[97m")

# print(f"dimenzije X {X_processed_new_for_prediction.shape} , dimenzije W {w_final.shape}")
print(
   f"PREDICTION FOR NEW VALUE: {np.exp(predict(normalize_new_data_point(X_processed_new_for_prediction, mu, sigma, columns_to_normalize), w_final, b_final))}")

# print(
#    f"PREDICTION FOR NEW VALUE: {predict(normalize_new_data_point(X_processed_new_for_prediction, mu, sigma, 7), w_final, b_final)}")