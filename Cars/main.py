import numpy as np
from normalization import reverse_zscore_normalize_features
from linearRegression import gradient_descent, compute_cost_linear_reg, compute_gradient_reg, predict
from preprocessing import get_data, fit_data_to_dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from normalization import zscore_normalize_features, normalize_new_data_point
from plots import plot_with_column, showHeatMap, showResult
import pandas as pd
import random

np.set_printoptions(threshold=np.inf)

df = get_data()

cols_to_plot_with_column = ["Mileage", "Power", "Year"]
cols_to_plot_heatmap = ["Price", "Mileage", "Power", "Year"]
#showHeatMap(df, cols_to_plot_heatmap)

# for i in range(len(cols_to_plot_with_column)):
#     plot_with_column(df, cols_to_plot_with_column[i])

y = df['Price']  # Target variable
df = df.drop('Price', axis=1)

df_for_predictions = pd.DataFrame(columns=df.columns)
row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
df_for_predictions = pd.concat([df_for_predictions, row_with_zeros], ignore_index=True)

columns_for_new_predictions = ['Manufacturer', 'Year', 'Mileage', 'Chasis', 'Fuel', 'Power', 'Emissions', 'Drivetrain', 'Transmission', 'Wheel side', 'Color', 'Registration',
                               'Damage']

new_row = {'Manufacturer': 'Audi', 'Year': 6, 'Mileage': 196707, 'Chasis': 'Džip/SUV', 'Fuel': 'Benzin', 'Power': 185,
           'Emissions': 'Euro 6', 'Drivetrain': '4x4', 'Transmission': 'Automatski / poluautomatski',
           'Wheel side': 'Levi volan', 'Color': 'Crna', 'Registration': 'Nije registrovan', 'Damage': 'Nije oštećen'}

X_new_for_prediction = pd.DataFrame([new_row], columns=columns_for_new_predictions)


X_processed_new_for_prediction = fit_data_to_dataset(df_for_predictions, X_new_for_prediction)
X_processed_new_for_prediction = X_processed_new_for_prediction.squeeze()

X = df
X = X.to_numpy()
y = y.to_numpy()

preprocessed_column_names = df.columns[:].tolist()
X, mu, sigma = zscore_normalize_features(X, 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

initial_w = np.array([-0.58687676,  0.03114216,  0.18585598, -0.19136025,  0.10205868,
  0.03742326, -0.05354726,  0.9188081 ,  1.05879028,  0.81747399,
  1.16012439, -0.13457126,  0.36669447,  0.16767429, -0.27845021,
 -0.10443184, -0.03635485, -0.19389015,  0.        , -0.00614297,
 -0.05683526, -0.12723187, -0.06376793,  0.27660243,  0.08883893,
  0.03039519,  0.44442957,  0.04544245, -0.15791805, -0.03156099,
  0.26330418,  0.31815749, -0.05601416,  0.20161527,  0.15846123,
  0.19756705,  0.05539662, -0.01321008,  0.28097395, -0.0822528 ,
 -0.04857932, -0.06245658,  0.08669825,  0.02421134,  0.        ,
  0.03269526,  0.0990155 ,  0.34261293,  0.34737129,  0.12849959,
 -0.64782839,  0.12243781,  0.44872831,  0.16006185,  0.39115873,
  0.03940494,  0.26033976,  0.21879092,  0.20268264,  0.25643123,
  0.14620314,  0.10896402,  0.23370977,  0.39829101,  0.51896081,
  0.57146962,  0.48070126,  0.6499189 ,  0.33813263,  0.50884558,
  0.61515374,  0.37464745,  0.50129788,  0.48649931,  0.07476414,
  0.12878859,  0.1069401 ,  0.14705804,  0.16562645,  0.09040789,
  0.18664245,  0.15757057, -0.06996381,  0.09084153,  0.09236039,
  0.15094353,  0.31633494,  0.11153195,  0.09306791,  0.02887327,
  0.09661509, -0.09504461,  0.10423996,  0.40937008,  0.41022282,
  0.50203258,  0.46048909,  0.19548381,  1.08432427,  0.22631121,
  0.6669629 ])
initial_b = 2.75
# initial_w = np.zeros(X_train.shape[1])
# initial_b = 0
iterations = 10
alpha = 5.0e-2
lambda_ = 1e0
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg,
                                            compute_gradient_reg, alpha, iterations, lambda_)

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
