import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from utils.smoothing import *
from utils.features import *


class TS_model():
    def __init__(self, freq_thresh = None, ts = None,  ts_name = None, 
                 lag = 7,type = 'BASE', prcnt = 0.85, window = 0, 
                 ema_span = 5, kalman_params = None,
                model_name = None,
                  metric = 'MAE_MAPE',
                  criterion = None, max_depth = None, min_samples_leaf = None, n_estimators = None,
                  gamma = None, learning_rate = None, alpha = None):
        self.freq_thresh = freq_thresh
        self.ts          = ts
        self.ts_name     = ts_name
        self.lag         = lag
        self.type        = type
        self.prcnt       = prcnt
        self.window      = window
        self.ema_span    = ema_span
        self.metric      = metric
        self.scaler      = StandardScaler()
        self.ts_smoothed = {}


        self.model_name         = model_name
        #Random Forest params
        self.criterion          = criterion
        self.max_depth          = max_depth
        self.min_samples_leaf   = min_samples_leaf
        self.n_estimators       = n_estimators


        #XGB params
        self.gamma              = gamma
        self.learning_rate      = learning_rate
        
        #Ridge
        self.alpha              = alpha

    def get_ts_smoothed(self):
        return self.ts_smoothed


    def create_data(self):
        if self.ts_name == 'SSE50':
            data = pd.read_csv("datasets/sse50_2.csv")

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

            data[["price"]] = data[["Price"]].map(convert_to_float)
            data["date"] = pd.to_datetime(data["Date"])
            data = data[["price", "date"]]          
        else:
            data = pd.read_excel("datasets/RUONIA.xlsx")
            data["date"] = pd.to_datetime(data["DT"])
            data = data[["date", "ruo"]]

        data = data[::-1]
        self.ts = data.reset_index(drop=True)
        return self.ts

    def create_lagged_data(self):
        if self.ts is None:
            self.create_data()

        if self.ts_name == 'SSE50':
            target_col = 'price'
        else:
            target_col = 'ruo'

        ts_values = self.ts[[target_col]]

        n = len(ts_values)
        lag = self.lag
        split_idx = int(n * self.prcnt)

        if n <= lag:
            raise ValueError(f"Длина временного ряда ({n}) меньше лага ({lag})")
        
        ts_train = ts_values[:split_idx]
        ts_test = ts_values[split_idx:]
        # scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(ts_train).squeeze()
        test_scaled  = self.scaler.transform(ts_test).squeeze()

        smooth_kwargs = {}
        smooth = None
        if   self.type == 'EMA':
            smooth        = apply_ema
            smooth_kwargs ={'ema_span' : self.ema_span}
            # train_scaled = apply_ema(train_scaled.squeeze(), self.ema_span)
            # test_scaled  = apply_ema(test_scaled.squeeze(), self.ema_span)
        elif self.type == 'KALMAN':
            smooth        = apply_kalman
            smooth_kwargs = {'params' : {}}
            # train_scaled = apply_kalman(train_scaled.squeeze())
            # test_scaled  = apply_kalman(test_scaled.squeeze())

        elif self.type == 'FOURIER':
            smooth        = apply_fourier
            smooth_kwargs = {'freq_thresh' : self.freq_thresh, 'window' : self.window}
            # train_scaled = apply_fourier(train_scaled.squeeze(), self.freq_thresh, self.window)
            # test_scaled  = apply_fourier(test_scaled.squeeze(),  self.freq_thresh, self.window)

        elif self.type == 'ALL':
            self.ts_smoothed['BASE']    = get_lag_features(train_scaled, test_scaled, lag)#[train_scaled.copy(), test_scaled.copy()]
            self.ts_smoothed['EMA']     = get_lag_features(train_scaled, test_scaled, lag, smooth=smooth, smooth_kwargs=smooth_kwargs)#[train_scaled.copy(), test_scaled.copy()]
            self.ts_smoothed['KALMAN']  = get_lag_features(train_scaled, test_scaled, lag, smooth=smooth, smooth_kwargs=smooth_kwargs)#[smooth(train_scaled.squeeze())['x'], smooth(test_scaled.squeeze())['x']]
            self.ts_smoothed['FOURIER']  = get_lag_features(train_scaled, test_scaled, lag, smooth=smooth, smooth_kwargs=smooth_kwargs)#[apply_kalman(train_scaled.squeeze())['x'], apply_kalman(test_scaled.squeeze())['x']]

        elif self.type != 'BASE':      
            raise ValueError(f"Не существующее значение параметра type")
        X_train, y_train, X_test, y_test = get_lag_features(train_scaled, test_scaled, lag, smooth=smooth, smooth_kwargs=smooth_kwargs)

        return X_train.squeeze(), y_train.squeeze(), X_test.squeeze(), y_test.squeeze()

    def select_model(self):
        print(self.model_name)


        if self.model_name == 'Ridge':
            model = Ridge(self.alpha)
        elif self.model_name == 'Random Forest':
            model = RandomForestRegressor(criterion=self.criterion, max_depth=self.max_depth, min_samples_leaf=2, n_estimators=self.n_estimators, random_state=52)
        elif self.model_name == 'XGB':
            model = XGBRegressor(max_depth = self.max_depth, learning_rate = self.learning_rate, gamma = self.gamma, n_estimators=self.n_estimators, random_state = 52)    
        else:
            raise ValueError(f"Неизвестное название модели")
        return model
    
    def select_metric(self):
        if self.metric == 'MAE':
            self.metric = mean_absolute_error
        elif self.metric == 'MAPE':
            self.metric = mean_absolute_percentage_error
        elif self.metric == 'MAE_MAPE':
            self.metric = [mean_absolute_error, mean_absolute_percentage_error]
        else:
            raise ValueError(f"Неизвестное название метрики")

    def fit_predict(self, X_train, y_train, X_test, y_test):
        # model = GradientBoostingRegressor(learning_rate=0.01, loss='absolute_error', max_depth=10, n_estimators=400, min_samples_leaf=3, random_state=52)
        model = self.select_model()
        self.select_metric()
        model.fit(X_train, y_train)

        y_pred_test    = model.predict(X_test)
        y_pred_train   = model.predict(X_train)
        
        # print("SHAPE", y_pred_test.shape)
        inv_pred_test  = self.scaler.inverse_transform(y_pred_test.reshape(-1, 1))
        inv_pred_train = self.scaler.inverse_transform(y_pred_train.reshape(-1, 1))

        inv_test  = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        inv_train = self.scaler.inverse_transform(y_train.reshape(-1, 1))


        res = {}
        if  len(self.metric) == 1:
            res[self.metric] = [self.metric(inv_pred_train, inv_train), self.metric(inv_pred_test,  inv_test)]
            # return { 'train' : self.metric(inv_pred_train, inv_train)*100.0,
            #     'test'  : self.metric(inv_pred_test,  inv_test)*100.0 }
        else:
            res = {}
            res['MAE'] = [self.metric[0](inv_pred_train, inv_train), self.metric[0](inv_pred_test, inv_test)]
            res['MAPE'] = [100*self.metric[1](inv_pred_train, inv_train), 100*self.metric[1](inv_pred_test, inv_test)]
            
        #load here
        return res

    def inv_scaler(self, x):
        return self.scaler.inverse_transform(x)



    def run(self):
        self.create_data()
        X_train, y_train, X_test, y_test = self.create_lagged_data()
        return self.fit_predict(X_train, y_train, X_test, y_test)
