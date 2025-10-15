import numpy as np

def get_lag_features(data_train, data_test, lag, smooth = None, smooth_kwargs = None, include_y = False):

    y_test_src = data_test[lag:]
    if smooth is not None:
        data_train = smooth(data_train, **smooth_kwargs)
        data_test  = smooth(data_test, **smooth_kwargs) 
    X_train = np.array([data_train[i:i+lag] for i in range(len(data_train) - lag)])
    y_train = data_train[lag:]
    X_test = np.array([data_test[i:i+lag] for i in range(len(data_test) - lag)])
    y_test = data_test[lag:]

    if include_y is False:
        y_test = y_test_src


    return X_train, y_train, X_test, y_test
