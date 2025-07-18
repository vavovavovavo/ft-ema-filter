import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


class TimeSeriesModel:
    def __init__(self, freq_thresh = 'auto', ts = None,  ts_name = None, lag = 7,type_ = ['EMA', 'FT', 'KALMAN', 'EMA+FT', 'BASE'], prcnt = 0.85, window = 0, ema_span = 5, kalman_params = None, models = LinearRegression, metric = mean_absolute_percentage_error):
        self.ts             = ts
        self.ts_name        = ts_name 
        self.type_          = type_
        self.prcnt          = prcnt
        self.freq_thresh    = freq_thresh
        self.window         = window
        self.ema_span       = ema_span
        self.scaler         = StandardScaler()
        self.kalman_params  = kalman_params if kalman_params is not None else {}
        self.models         = models
        self.metric         = metric
        self.lag            = lag

    def create_ts(self):
        if self.ts_name == 'SSE50':
            data  = pd.read_csv("../datasets/sse50_2.csv")
            def convert_to_float(value):
                if isinstance(value, str):  # Если значение строка
                    value = value.replace(',', '')  # Удаляем запятые
                    if 'B' in value:  # Обработка миллиардов
                        return float(value.replace('B', '')) * 1e9
                    elif 'M' in value:  # Обработка миллионов
                        return float(value.replace('M', '')) * 1e6
                    elif '%' in value:  # Обработка процентов
                        return float(value.replace('%', '')) / 100
                    else:
                        return float(value)  # Просто преобразуем в float
                return float(value)  # Если значение уже числовое

            # data = data[::-1]
            data[["Price"]] = data[["Price"]].applymap(convert_to_float)
            data["Date"] = pd.to_datetime(data["Date"])
            data = data[["Price", "Date"]]
            data = data[::-1]
            data = data.reset_index(drop=True)
        elif self.ts_name == 'RUO':
            data    = pd.read_excel("../datasets/RUONIA.xlsx")
            data = data[["DT", "ruo"]]
            data["DT"] = pd.to_datetime(data["DT"])
            data = data.iloc[::-1]
            data = data.reset_index(drop=True)
        self.ts = data

    def create_lagged_dataset(self, values, real, lag_length):
        if isinstance(values, pd.Series):
            values = values.values
        x = np.array([values[i-lag_length:i] for i in range(lag_length, len(values))])
        y = real[lag_length:]
        return x, y
    def preprocess_data(self, data):
        return self.scaler.fit_transform(data.reshape(-1, 1))
    def apply_ema(self, series):
        return series.ewm(span=self.ema_span, adjust=False).mean().values
    def apply_fourier_transform(self, series, freq_thresh, window):
        my_series = series.copy().astype(float)
        y_f = rfft(my_series)
        h = 1.0 / 365.0
        x_f = rfftfreq(n=len(my_series), d=h)
        y_f[x_f > freq_thresh] = 0
        fourier_vals = irfft(y_f, n=len(my_series))
        if window == 0:
            result = fourier_vals
        else:
            result = np.concatenate([my_series[:window], fourier_vals[window:-window], my_series[-window:]])
        return result
    def annealing(self, data):
        mse = lambda theta :  mean_squared_error(data, self.apply_fourier_transform(data, theta, self.window))
        x0 = [20.0]
        minimizer_kwargs = {"method": "BFGS"}
        ret = basinhopping(mse, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
        return ret.x
    def fit_and_predict(self, X_train, y_train, X_test, y_test, new_metric = None):
        model = self.models()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        inv_pred_train = self.scaler.inverse_transform(model.predict(X_train).reshape(-1, 1))
        inv_train = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        inv_pred_test = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        inv_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        if new_metric == None:
            return self.metric(inv_pred_train, inv_train), self.metric(inv_pred_test, inv_test) 
        else:
            return new_metric(inv_pred_train, inv_train), new_metric(inv_pred_test, inv_test)
    def kalman_filter(self, data, params = None):
        n = len(data)
        x_estimates = np.zeros((n, 1))
        P_estimates = np.zeros((n, 1))
        F      = self.kalman_params.get('F', np.eye(1))
        H      = self.kalman_params.get('H', np.eye(1))
        R      = self.kalman_params.get('R', np.eye(1))
        Q      = self.kalman_params.get('Q', np.eye(1))
        x_pred = self.kalman_params.get('x0', np.zeros((1, 1)))
        P_pred = self.kalman_params.get('P0', np.eye(1))
        if params:
            if 'F' in params: F = params['F']
            if 'H' in params: H = params['H']
            if 'R' in params: R = params['R']
            if 'Q' in params: Q = params['Q']
            if 'x0' in params: x_pred = params['x0']
            if 'P0' in params: P_pred = params['P0']
        for i in range(n):
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x_filt = x_pred + K @ (data[i] - H @ x_pred)
            P_filt = (np.eye(1) - K @ H) @ P_pred
            x_estimates[i] = x_filt.flatten()
            P_estimates[i] = P_filt.flatten()
            x_pred = F @ x_filt  
            P_pred = F @ P_filt @ F.T + Q  
        return { 'x' : x_estimates.flatten(), 'P' : P_estimates.flatten()}
    def data_prep(self):
        data   = {method: [] for method in self.type_}
        training_set = self.ts.iloc[:int(self.prcnt * self.ts.shape[0])]
        test_set     = self.ts.iloc[int(self.prcnt * self.ts.shape[0]):]
        training_set_scaled = self.scaler.fit_transform(training_set.values.reshape(-1, 1))
        test_set_scaled     = self.scaler.transform(test_set.values.reshape(-1, 1))
        if "BASE" in self.type_:
            X_train, y_train = self.create_lagged_dataset(training_set_scaled.reshape(-1, 1).flatten(), training_set_scaled.reshape(-1, 1).flatten(), self.lag)
            X_test, y_test   = self.create_lagged_dataset(test_set_scaled.reshape(-1, 1).flatten(), test_set_scaled.reshape(-1, 1).flatten(), self.lag)
            data['BASE'] =  [X_train, y_train, X_test, y_test]
        if "EMA" in self.type_:
            ema_values_train = self.apply_ema(pd.Series(training_set_scaled.reshape(-1, 1).flatten()))
            ema_values_test  = self.apply_ema(pd.Series(test_set_scaled.reshape(-1, 1).flatten()))
            X_train, y_train = self.create_lagged_dataset(ema_values_train, training_set_scaled, self.lag)
            X_test, y_test   = self.create_lagged_dataset(ema_values_test, test_set_scaled, self.lag)
            data['EMA'] =  [X_train, y_train, X_test, y_test]
        if "KALMAN" in self.type_:
            kalman_filtered_train = self.kalman_filter(training_set_scaled.copy())['x']
            kalman_filtered_test  = self.kalman_filter(test_set_scaled.copy())['x']
            X_train, y_train = self.create_lagged_dataset(kalman_filtered_train, training_set_scaled, self.lag)
            X_test, y_test = self.create_lagged_dataset(kalman_filtered_test, test_set_scaled, self.lag)
            data['KALMAN'] = [X_train, y_train, X_test, y_test]
        if  self.freq_thresh !='auto':   
            for freq_thresh_ in self.freq_thresh:
                if "FT" in self.type_:
                    fourier_vals_train = self.apply_fourier_transform(training_set_scaled.reshape(1, -1).flatten(), freq_thresh_, self.window)
                    fourier_vals_test  = self.apply_fourier_transform(test_set_scaled.reshape(1, -1).flatten(), freq_thresh_, self.window)
                    X_train, y_train = self.create_lagged_dataset(fourier_vals_train, training_set_scaled, self.lag)
                    X_test, y_test   = self.create_lagged_dataset(fourier_vals_test, test_set_scaled, self.lag)
                    data['FT'].append([X_train, y_train, X_test, y_test])
                if "EMA+FT" in self.type_:
                    ema_values_train = self.apply_ema(pd.Series(training_set_scaled.reshape(-1, 1).flatten()))
                    ema_values_test  = self.apply_ema(pd.Series(test_set_scaled.reshape(-1, 1).flatten()))
                    fourier_vals_train = self.apply_fourier_transform(ema_values_train, freq_thresh_, self.window)
                    fourier_vals_test = self.apply_fourier_transform(ema_values_test, freq_thresh_, self.window)
                    X_train, y_train = self.create_lagged_dataset(fourier_vals_train, training_set_scaled, self.lag)
                    X_test, y_test = self.create_lagged_dataset(fourier_vals_test, test_set_scaled, self.lag)
                    data['EMA+FT'].append([X_train, y_train, X_test, y_test])
        else:
            if "FT" in self.type_:
                freq_thresh = self.annealing(training_set_scaled.reshape(-1))
                fourier_vals_train = self.apply_fourier_transform(training_set_scaled.reshape(-1), freq_thresh, self.window)
                fourier_vals_test  = self.apply_fourier_transform(test_set_scaled.reshape(-1), freq_thresh, self.window)
                X_train, y_train = self.create_lagged_dataset(fourier_vals_train, training_set_scaled, self.lag)
                X_test, y_test   = self.create_lagged_dataset(fourier_vals_test, test_set_scaled, self.lag)
                data['FT'].append([X_train, y_train, X_test, y_test])
            if "EMA+FT" in self.type_:
                ema_values_train = self.apply_ema(pd.Series(training_set_scaled.reshape(-1, 1).flatten()))
                ema_values_test  = self.apply_ema(pd.Series(test_set_scaled.reshape(-1, 1).flatten()))
                freq_thresh = self.annealing(ema_values_train)
                fourier_vals_train = self.apply_fourier_transform(ema_values_train, freq_thresh, self.window)
                fourier_vals_test = self.apply_fourier_transform(ema_values_test, freq_thresh, self.window)
                X_train, y_train = self.create_lagged_dataset(fourier_vals_train, training_set_scaled, self.lag)
                X_test, y_test = self.create_lagged_dataset(fourier_vals_test, test_set_scaled, self.lag)
                data['EMA+FT'].append([X_train, y_train, X_test, y_test])
        return data
    def run(self, new_metric = None):
        result_mae   = {method: [] for method in self.type_}
        data = self.data_prep()
        if "BASE" in self.type_:
            X_train, y_train, X_test, y_test = data['BASE']
            result_mae["BASE"] = self.fit_and_predict(X_train, y_train, X_test, y_test, new_metric)
        if "EMA" in self.type_:
            X_train, y_train, X_test, y_test = data['EMA']
            result_mae["EMA"] = self.fit_and_predict(X_train, y_train, X_test, y_test, new_metric)
        if "KALMAN" in self.type_:
            X_train, y_train, X_test, y_test = data['KALMAN']
            result_mae["KALMAN"] = self.fit_and_predict(X_train, y_train, X_test, y_test, new_metric)
        for i in range(len(self.freq_thresh)):
            if "FT" in self.type_:
                X_train, y_train, X_test, y_test = data['FT'][i]
                result_mae["FT"].append(self.fit_and_predict(X_train, y_train, X_test, y_test, new_metric))
            if "EMA+FT" in self.type_:
                X_train, y_train, X_test, y_test = data['EMA+FT'][i]
                result_mae["EMA+FT"].append(self.fit_and_predict(X_train, y_train, X_test, y_test, new_metric))
        return result_mae