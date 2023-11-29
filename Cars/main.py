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

new_row = {'Manufacturer': 'Volkswagen', 'Year': 17, 'Mileage': 260000, 'Chasis': 'Hečbek', 'Fuel': 'Dizel', 'Power': 77,
           'Emissions': 'Euro 4', 'Displacement': 1896, 'Drivetrain': 'Prednji', 'Transmission': 'Manuelni 6 brzina',
           'Wheel side': 'Levi volan', 'Color': 'Plava', 'Registration': 'Registrovan', 'Damage': 'Nije oštećen'}
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

initial_w = np.array([-0.57366918, -0.00930916,  0.00991392,  0.13729093, -0.18005065,
  0.12296758,  0.00685235, -0.02674028, -0.01626193,  0.03576699,
  0.97718443,  1.10916755,  0.9460232 ,  1.14032878, -0.09563446,
  0.45153296,  0.21943507, -0.30643948, -0.20482194,  0.02502704,
 -0.19196948, -0.00674629, -0.02369257, -0.04957895, -0.14013788,
  0.0377333 ,  0.20246919, -0.01234677, -0.0058266 ,  0.62187652,
  0.0513633 ,  0.06655301, -0.04351597,  0.34351654,  0.12189803,
 -0.03021448,  0.27536171,  0.23113651,  0.09757473,  0.1131279 ,
  0.01427556,  0.        , -0.04331977, -0.00833331, -0.0545957 ,
  0.17385667, -0.19049571, -0.0228821 ,  0.        ,  0.17629877,
  0.35557554,  0.37747329,  0.12225677, -0.71312338,  0.1516844 ,
  0.42733961,  0.18330831,  0.4433601 ,  0.07075488,  0.30374777,
  0.2307858 ,  0.16755949,  0.25949602,  0.15658683,  0.14164698,
  0.23097763,  0.39351288,  0.55261246,  0.6110152 ,  0.50085472,
  0.66126405,  0.36651055,  0.55772267,  0.62916596,  0.42766441,
  0.49644682,  0.53307479,  0.0651212 ,  0.20851904,  0.1115433 ,
  0.17113121,  0.15841965,  0.08780252,  0.19922129,  0.        ,
 -0.13980018,  0.05937291,  0.14620471,  0.14931123,  0.12146341,
  0.08863667,  0.11094464,  0.11910859,  0.11088937,  0.21478017,
  0.10368225,  0.47525271,  0.49712886,  0.59658623,  0.51738418,
  1.07808441,  0.31731853,  0.69094904])
initial_b = 2.55
# initial_w = np.zeros(X_train.shape[1])
# initial_b = 0
iterations = 10
alpha = 5.0e-2
lambda_ = 0.5e0
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
