import copy
import math

import numpy as np


# Predict without vectors (for loop)
def predict_single_loop(x, w, b):
    """
    Single predict using linear regression
    Parameters:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p


# Predict with vectors
def predict(x, w, b):
    """
    Single predict using linear regression
    Parameters:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p


def compute_vec_cost(X, y, w, b):
    """
    Vectorized version of compute_cost function
    Parameters:
       X (ndarray (m,n)): Data, m examples with n features
       y (ndarray (m,)) : target values
       w (ndarray (n,)) : model parameters
       b (scalar)       : model parameter
    Returns:
       cost (scalar): cost
    """
    m = X.shape[0]
    f_wb = np.dot(X, w) + b
    cost = np.sum((f_wb - y) ** 2) / (2 * m)
    return cost


# Compute gradient, return dj_db and dj_dw
def compute_gradient_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression with regularization
    Parameters:
      X : (ndarray (m,n)) Data, m examples with n features
      y : (ndarray (m,)) target values
      w : (ndarray (n,)) model parameters
      b : (scalar) model parameter
      lambda_: (scalar) regularization parameter
    Returns:
      dj_dw : (ndarray (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar) The gradient of the cost w.r.t. the parameter b.     
    """
    m, n = X.shape  # (number of examples, number of features)

    err = np.dot(X, w) + b - y  # shape (m,)
    dj_dw = np.dot(err, X) / m + (lambda_ / m) * w  # shape (n,)
    dj_db = np.sum(err) / m

    return dj_db, dj_dw


# Compute gradient, return dj_db and dj_dw
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Parameters:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m = X.shape[0]  # number of examples
    y_hat = np.dot(X, w) + b  # predicted values

    err = y_hat - y

    dj_dw = np.dot(X.T, err) / m
    dj_db = err.mean()

    return dj_db, dj_dw


def compute_cost_linear_reg(X, y, w, b, lambda_):
    """
    Computes the cost over all examples
    Parameters:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost
    """
    m = X.shape[0]
    n = len(w)

    X_b = np.c_[X, np.ones(m)]  # augmented feature vectors
    w_b = np.r_[w, b]  # augmented parameter vector

    cost = np.sum((np.dot(X_b, w_b) - y) ** 2) / (2 * m)
    reg_cost = lambda_ * np.sum(w_b[:-1] ** 2) / (2 * m)

    total_cost = cost + reg_cost
    return total_cost


# Run gradient descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha
    Parameters:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b, lambda_))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.6f}   ")

    return w, b, J_history  # return final w,b and J history for graphing
