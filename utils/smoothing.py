import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft

def apply_ema(series, ema_span):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    return series.ewm(span=ema_span, adjust=False).mean().values

def apply_fourier(series, freq_thresh, window):
    if isinstance(series, pd.Series):
        series = series.values
    
    y_f = rfft(series)
    h = 1.0 / 365.0
    x_f = rfftfreq(n=len(series), d=h)
    y_f[x_f > freq_thresh] = 0
    fourier_vals = irfft(y_f, n=len(series))
    
    if window == 0:
        result = fourier_vals
    else:
        result = np.concatenate([series[:window], fourier_vals[window:-window], series[-window:]])
    return result   
    
def apply_kalman(series, params = {}):
    if isinstance(series, pd.Series):
        series = series.values
    
    n = len(series)
    x_estimates = np.zeros((n, 1))
    P_estimates = np.zeros((n, 1))
    F      = params.get('F', np.eye(1))
    H      = params.get('H', np.eye(1))
    R      = params.get('R', np.eye(1))
    Q      = params.get('Q', np.eye(1))
    x_pred = params.get('x0', np.zeros((1, 1)))
    P_pred = params.get('P0', np.eye(1))  

    for i in range(n):
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_filt = x_pred + K @ (series[i] - H @ x_pred)
        P_filt = (np.eye(1) - K @ H) @ P_pred
        x_estimates[i] = x_filt.flatten()
        P_estimates[i] = P_filt.flatten()
        x_pred = F @ x_filt  
        P_pred = F @ P_filt @ F.T + Q  
    return x_estimates.flatten()