import numpy as np
from normalization import reverse_zscore_normalize_features
from linearRegression import gradient_descent, compute_cost_linear_reg, compute_gradient_reg, predict
from preprocessing import get_data, fit_data_to_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from normalization import zscore_normalize_features, normalize_new_data_point
from plots import plot_with_column, showHeatMap
import pandas as pd

np.set_printoptions(threshold=np.inf)

df = get_data()

cols_to_plot_with_column = ["Mileage", "Power", "Year"]
cols_to_plot_heatmap = ["Price", "Mileage", "Power", "Year"]
# showHeatMap(df, cols_to_plot_heatmap)

# for i in range(len(cols_to_plot_with_column)):
#    plot_with_column(df,cols_to_plot_with_column[i],)

y = df['Price']  # Target variable
df = df.drop('Price', axis=1)

df_for_predictions = pd.DataFrame(columns=df.columns)
row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df_for_predictions = pd.concat([df_for_predictions, row_with_zeros], ignore_index=True)

columns_for_new_predictions = ['Manufacturer', 'Year', 'Mileage', 'Chasis', 'Fuel', 'Power', 'Emissions',
                               'Displacement', 'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration',
                               'Damage']
X_new_for_prediction = pd.DataFrame(columns=columns_for_new_predictions)

new_row = {'Manufacturer': 'BMW', 'Year': 10, 'Mileage': 230000, 'Chasis': 'Hečbek', 'Fuel': 'Dizel', 'Power': 85,
           'Emissions': 'Euro 5', 'Displacement': 1995, 'Drivetrain': 'Zadnji', 'Transmission': 'Manuelni 6 brzina',
           'Wheel side': 'Levi volan', 'Color': 'Crna', 'Registration': 'Nije registrovan', 'Damage': 'Nije oštećen'}
df_new_row = pd.DataFrame([new_row])

X_new_for_prediction = pd.concat([X_new_for_prediction, df_new_row], ignore_index=True)

X_processed_new_for_prediction = fit_data_to_dataset(df_for_predictions, X_new_for_prediction)
X_processed_new_for_prediction = X_processed_new_for_prediction.squeeze()

X = df
X = X.to_numpy()
y = y.to_numpy()

preprocessed_column_names = df.columns[:].tolist()
X, mu, sigma = zscore_normalize_features(X, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 1000
alpha = 5.0e-2
lambda_ = 1e0
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg,
                                            compute_gradient_reg, alpha, iterations, lambda_)

# For displaying purposes
X_train_reversed = reverse_zscore_normalize_features(X_train, mu, sigma, 10)
X_test_reversed = reverse_zscore_normalize_features(X_test, mu, sigma, 10)

train_percentage = []
y_predicted_train = np.array([])
for i in range(X_train_reversed.shape[0]):
    row = np.exp(np.dot(X_train[i], w_final) + b_final)
    y_train_reversed = np.exp(y_train[i])
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

# for i in range(0, x_train_normalized.shape[1]):
#    plt.scatter(x_train_normalized[:,i], y_train, marker='x', c='r', label="Actual Value"); plt.title(f"Price/{cols_[i]}")
#    plt.scatter(x_train_normalized[:,i], np.dot(x_train_eg_normalized[:,i], w_final[i]) + b_final, label="Predicted Value", marker='o',c='b'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print(f"\033[97mb,w found by gradient descent: {b_final:0.2f},{np.array2string(w_final, separator=', ')}\033[97m")

for i in range(len(preprocessed_column_names)):
    print(f"Weight for '{preprocessed_column_names[i]}': {w_final[i]:0.2f}")

# Plot the correlation
# for i in range(0,x_train_normalized.shape[1]):
#    plot_with_column(x_train_normalized, y_train, i, df.columns[i], f"Price/{df.columns[i]}")

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

print(f"dimenzije X {X_processed_new_for_prediction.shape} , dimenzije W {w_final.shape}")
print(
    f"PREDICTION FOR NEW VALUE: {np.exp(predict(normalize_new_data_point(X_processed_new_for_prediction, mu, sigma), w_final, b_final))}")
