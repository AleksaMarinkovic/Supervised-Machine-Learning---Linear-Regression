import numpy as np
from plots import plot_with_column
from normalization import reverse_zscore_normalize_features
from linearRegression import compute_gradient, gradient_descent, compute_cost_linear_reg
from preprocessing import get_train_and_test_data, plotHistograms, plotBarGraphs
import pandas as pd
np.set_printoptions(threshold=np.inf)

X_train, X_test, y_train, y_test, mu_train, sigma_train, mu_test, sigma_test, X_original, y_original, original_column_names, preprocessed_column_names = get_train_and_test_data()
plotHistograms(pd.concat([X_original, y_original], ignore_index=True), 2,2,["Year","Mileage","Power","Price"],["Year","KM","KW","Euros"])
plotBarGraphs(X_original,5,3, ['Manufacturer', 'Chasis' ,'Fuel type', 'Emissions', 'Drivetrain', 'Transmission', 'Number of doors', 'Number of seats', 'Wheel side','Air conditioning', 'Color', 'Registration', 'Damage'])
# initial_w = np.array([ 2429.75555231,  -119.04228613,  1465.90150984,  -406.51050697,
#   1443.71331652,  -749.95538199,  -274.79170187,   -82.6006433 ,
#   -311.05303765,  -473.63142805,   -28.3923123 ,  -259.61619587,
#   -367.16196176,   175.91610234,    61.07064587,   -80.33332051,
#    423.83834122,    24.37059824,  -263.25681912,   -70.98256535,
#   1165.84267524,   133.03400222,    94.80217543,   618.05080607,
#     84.38889476,   912.10012888,   -24.15279825,  -941.96587986,
#    498.53861599,  -592.43287383,    96.46168487,  -154.51942276,
#   -133.7126114 ,  -332.33894468,  -215.60562004,  -185.88076556,
#    629.49775368,   707.75903416,  -611.81803797,  -116.80646527,
#    -80.08774989,  2207.63675461, -1094.14931058,  1015.02261636,
#  -1275.97886895,  -267.10942696,  -526.78583235,  -145.65684886,
#    398.79865479,  -251.32284892,    92.9404984 ,  -780.93351298,
#    132.91837293,  -427.65393227,  1545.82915882,   292.84030278,
#    444.93626827,  -517.38601078,    91.38717209,  1731.93993616,
#   -363.93179707,  -111.01327073,  -945.21713109,   427.0275556 ,
#   -115.24982313,  -472.05983177,    26.8166494 ,    58.31959132,
#    391.08196881,    57.59253057,   313.10479367,   -63.07796943,
#     -7.97163992,   267.69602876,    52.05334345,  -776.8082042 ,
#     83.8972752 ,  -123.97526801,   290.10844779,  -576.12432288,
#   -691.41347924,  -665.11135696,   -61.27435161,  -101.70631991,
#      0.        ,   745.66136702,   461.24556206,   380.92232251,
#   1160.48296394,   296.17322777,  -299.50998138,    74.29715792,
#    143.86757273,   -28.95488148,   -88.9561335 ,  -234.28081602,
#    -82.26600699,   577.41952611,  -676.77900413,   816.64016672,
#   1641.58880982,  -683.82411958,   662.29941643,   -73.94333409,
#   -645.98695797,  1971.28850017,   496.00499313,  1299.99317116,
#    -40.20091102])
initial_w = np.zeros(X_train.shape[1])
initial_b = 0
iterations = 1000
alpha = 1.0e-3
lambda_ = 1.0e0
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost_linear_reg, compute_gradient, alpha, iterations, lambda_)

# For displaying purposes
X_train_reversed = reverse_zscore_normalize_features(X_train,mu_train, sigma_train, 3)
X_test_reversed = reverse_zscore_normalize_features(X_test,mu_test, sigma_test, 3)

train_percentage = []
for i in range(X_train_reversed.shape[0]):
    y_predicted_train = np.dot(X_train[i], w_final) + b_final
    print(f"\033[93mtrain {i} prediction: {y_predicted_train:0.2f}, target value: {y_train[i]}, year: {X_train_reversed[i,0]}, mileage: {X_train_reversed[i,1]}, power: {X_train_reversed[i,2]}\033[93m")
    individual_train_percentage = (y_train[i]*100)/y_predicted_train if y_train[i] < y_predicted_train else (y_predicted_train*100)/y_train[i]
    train_percentage.append(individual_train_percentage)

test_percentage = []
for i in range(X_test_reversed.shape[0]):
    y_predicted_test = np.dot(X_test[i], w_final) + b_final
    print(f"\033[94mtest {i} prediction: {y_predicted_test:0.2f}, target value: {y_test[i]}, year: {X_test_reversed[i,0]}, mileage: {X_test_reversed[i,1]}, power: {X_test_reversed[i,2]}\033[94m")
    individual_test_percentage = (y_test[i]*100)/y_predicted_test if y_test[i] < y_predicted_test else (y_predicted_test*100)/y_test[i]
    test_percentage.append(individual_test_percentage)

#for i in range(0, x_train_normalized.shape[1]):
#    plt.scatter(x_train_normalized[:,i], y_train, marker='x', c='r', label="Actual Value"); plt.title(f"Price/{cols_[i]}")
#    plt.scatter(x_train_normalized[:,i], np.dot(x_train_eg_normalized[:,i], w_final[i]) + b_final, label="Predicted Value", marker='o',c='b'); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print(f"\033[97mb,w found by gradient descent: {b_final:0.2f},{np.array2string(w_final, separator=', ')}\033[97m")
for i in range(len(preprocessed_column_names)):
    print(f"Weight for '{preprocessed_column_names[i]}': {w_final[i]:0.2f}")

#Plot the correlation
#for i in range(0,x_train_normalized.shape[1]):
#    plot_with_column(x_train_normalized, y_train, i, df.columns[i], f"Price/{df.columns[i]}")

print("\n")
print("\033[93mTRAIN\033[93m \033[92mMEAN PERCENTAGE: \033[92m")
print(np.mean(train_percentage))
print("\033[94mTEST\033[94m \033[92mMEAN PERCENTAGE: \033[92m")
print(np.mean(test_percentage))
print("\033[97m")