import numpy as np

def zscore_normalize_features(X, num_cols=5):
    """
    Computes X, z-score normalized by column for the first 'num_cols' columns.
    
    Args:
      X (ndarray (m,n)): input data, m examples, n features
      num_cols (int): number of columns to normalize
    
    Returns:
      X_norm (ndarray (m,n)): input normalized by column for the first 'num_cols' columns
      mu (ndarray (n,)): mean of each feature for the first 'num_cols' columns
      sigma (ndarray (n,)): standard deviation of each feature for the first 'num_cols' columns
    """
     # find the mean of each column/feature for the first 'num_cols' columns
    mu = np.mean(X[:, :num_cols], axis=0)  # mu will have shape (num_cols,)
    # find the standard deviation of each column/feature for the first 'num_cols' columns
    sigma = np.std(X[:, :num_cols], axis=0)  # sigma will have shape (num_cols,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = np.concatenate([(X[:, :num_cols] - mu) / sigma, X[:, num_cols:]], axis=1)

    return X_norm, mu, sigma


def reverse_zscore_normalize_features(X_norm, mu, sigma, num_cols=5):
    """
    Reverts z-score normalization for the first 'num_cols' columns using mean and standard deviation.

    Args:
      X_norm (ndarray (m,n)): z-score normalized data, m examples, n features
      mu (ndarray (n,)): mean of each feature for the first 'num_cols' columns
      sigma (ndarray (n,)): standard deviation of each feature for the first 'num_cols' columns
      num_cols (int): number of columns to revert normalization

    Returns:
      X_original (ndarray (m,n)): original data for the first 'num_cols' columns
    """
    # Reverse normalization: multiply by std and add back the mean
    X_original = X_norm.copy()  # Avoid modifying the original array
    X_original[:, :num_cols] = X_original[:, :num_cols] * sigma + mu

    return X_original


def zscore_normalize_target(y):
    """
    computes  Y, zcore normalized
    
    Args:
      y (ndarray (m,))     : input data, m examples
      
    Returns:
      y_norm (ndarray (m,n)): input normalized by column
      mu_Y (ndarray (n,))     : mean of each feature
      sigma_Y (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu_Y     = np.mean(y)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma_Y  = np.std(y)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    y_norm = (y - mu_Y) / sigma_Y      

    return (y_norm, mu_Y, sigma_Y)