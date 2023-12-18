import numpy as np
from normalization import reverse_zscore_normalize_features
from linear_regression import gradient_descent, compute_cost_linear_reg, compute_gradient_reg, predict
from preprocessing import get_data, fit_data_to_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from normalization import zscore_normalize_features, normalize_new_data_point
from plots import plot_with_column, showHeatMap, showResult
import pandas as pd
import random
import time
np.set_printoptions(threshold=np.inf)

df = get_data()

cols_to_plot_with_column = ["Mileage", "Power", "Year"]
cols_to_plot_heatmap = ["Price", "Mileage", "Power", "Year"]
#showHeatMap(df, cols_to_plot_heatmap)

for i in range(len(cols_to_plot_with_column)):
    plot_with_column(df, cols_to_plot_with_column[i])

y = df['Price']  # Target variable
df = df.drop('Price', axis=1)

df_for_predictions = pd.DataFrame(columns=df.columns)
row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df_for_predictions = pd.concat([df_for_predictions, row_with_zeros], ignore_index=True)

columns_for_new_predictions = ['Manufacturer', 'Year', 'Mileage', 'Chasis', 'Fuel', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration',
                               'Damage']

new_row = {'Manufacturer': 'Honda', 'Year': 7, 'Mileage': 50000, 'Chasis': 'Džip/SUV', 'Fuel': 'Dizel', 'Power': 118,
           'Emissions': 'Euro 6', 'Drivetrain': '4x4', 'Transmission': 'Automatski / poluautomatski',
           'Wheel side': 'Levi volan', 'Color': 'Bela', 'Registration': 'Registrovan', 'Damage': 'Nije oštećen'}

X_new_for_prediction = pd.DataFrame([new_row], columns=columns_for_new_predictions)


X_processed_new_for_prediction = fit_data_to_dataset(df_for_predictions, X_new_for_prediction)
X_processed_new_for_prediction = X_processed_new_for_prediction.squeeze()

X = df
X = X.to_numpy()
y = y.to_numpy()

preprocessed_column_names = df.columns[:].tolist()
X, mu, sigma = zscore_normalize_features(X, 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

# initial_w = np.array([-6.10380237e-01,  7.64766537e-02,  2.23190178e-01, -2.89012067e-01,
#                       1.58842058e-01, -7.31995823e-02,  6.31418691e-03,  2.91957136e-01,
#                       4.60018390e-01,  2.33922439e-01,  5.18053088e-01, -1.47266830e-01,
#                       3.87361675e-01,  1.85098347e-01, -5.65151818e-01, -1.90625453e-01,
#                       -2.10586730e-02, -3.23192186e-01, -6.56614059e-02, -8.92549276e-03,
#                       -9.40414706e-02, -1.50215534e-01, -7.80138585e-02,  2.19189655e-01,
#                       -2.94748606e-02,  3.92098189e-02,  4.58295969e-01,  4.03193613e-02,
#                       -3.16905815e-01, -3.42975596e-02,  3.11080620e-01,  2.89933728e-01,
#                       -8.34130685e-02,  2.08003692e-01,  1.19246141e-01,  1.58387992e-01,
#                       5.82624629e-02, -2.84480710e-02,  5.28815729e-01, -1.16245774e-01,
#                       -4.23003632e-02, -1.99548301e-03,  3.44030562e-02, -1.59466115e-01,
#                       -8.53635501e-02,  1.18979408e-01,  1.10631506e-02,  2.50907279e-01,
#                       3.25494277e-01,  1.26202254e-01, -6.90304062e-01,  1.14088356e-01,
#                       2.83701473e-01, -3.40752334e-03,  2.35056214e-01, -9.62923434e-02,
#                       7.41562016e-02,  6.41155091e-02,  2.22830550e-02,  1.72362941e-01,
#                       -1.04712095e-01, -1.50137133e-02,  3.71703197e-02,  1.86429887e-01,
#                       3.09285602e-01,  3.38815525e-01,  1.62334097e-01,  3.76586683e-01,
#                       -4.93406679e-04,  2.13548152e-01,  2.33618581e-01,  2.73864063e-01,
#                       1.21238747e-01,  1.23254136e-01, -2.77843989e-02, -4.72284781e-02,
#                       1.12004965e-01,  5.10769817e-02,  3.51772637e-02, -1.79853155e-02,
#                       1.10859637e-01,  3.13348918e-01, -1.13660951e-01,  3.96000428e-02,
#                       -2.05869505e-03,  1.13103428e-02,  2.47410969e-01,  1.56815472e-02,
#                       -1.30689757e-02,  3.13251080e-02, -2.81239095e-02,  1.68918512e-03,
#                       3.24012903e-02,  1.06239375e-01,  6.80926104e-02,  2.51058855e-01,
#                       1.46631228e-01,  1.79953459e-01,  6.44882056e-01, -1.72689517e-01,
#                       2.79782988e-01])
# initial_b = 5.95
initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 10000
alpha = 5.0e-2
lambda_ = 1e0
start_time = time.time()
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg,
                                            compute_gradient_reg, alpha, iterations, lambda_)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Gradient descent took: {elapsed_time} seconds")
# For displaying purposes
X_train_reversed = reverse_zscore_normalize_features(X_train, mu, sigma, 7)
X_test_reversed = reverse_zscore_normalize_features(X_test, mu, sigma, 7)

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
#showResult(X_test_reversed, np.exp(y_test), y_predicted_test)

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
    f"PREDICTION FOR NEW VALUE: {np.exp(predict(normalize_new_data_point(X_processed_new_for_prediction, mu, sigma, 7), w_final, b_final))}")
