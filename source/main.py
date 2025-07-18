from TS_model import TimeSeriesModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import argparse
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Time Series Model Runner")
    parser.add_argument('--prcnt', type=float, default=0.85, help='Train/test split ratio')
    parser.add_argument('--freq_thresh', type=str, default='auto', help='Frequency threshold(s) for Fourier Transform, comma-separated')
    parser.add_argument('--type_', type=str, default='[BASE,EMA,KALMAN,FT,EMA+FT]', help='Model types to run, comma-separated')
    parser.add_argument('--lag', type=int, default=10, help='Lag length for lagged dataset')
    parser.add_argument('--window', type=int, default=0, help='Window size for Fourier transform')
    parser.add_argument('--ema_span', type=int, default=7, help='Span for EMA')
    parser.add_argument('--ts_name', type=str, default='SSE50', help='Name of CSV file with time series')
    parser.add_argument('--kalman_params', type=str, default='default', help='Params of kalamn equations')
    parser.add_argument('--metric', type=str, default='mse', help='Metric of quality')
    parser.add_argument('--models', type=str, default='LR', help='Models of quality')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Parse freq_thresh
    if args.freq_thresh == 'auto':
        freq_thresh = 'auto'
    else:
        freq_thresh = list(map(float, args.freq_thresh.split(',')))

    # Parse type_
    type_list = args.type_.split(',')


    # Initialize and run model
    metric = None
    if args.metric == 'mape':
        metric = mean_absolute_percentage_error
    elif args.metric == 'mse':
        metric = mean_squared_error
    else:
        metric = mean_absolute_percentage_error

    models = None
    if args.models == 'LR':
        models = LinearRegression
    elif args.models == 'XGB':
        models = GradientBoostingRegressor(learning_rate= 0.1, loss= 'huber', n_estimators= 300)
    else:
        models = RandomForestRegressor(criterion= 'squared_error', max_depth= 10, min_samples_split =  5, n_estimators = 100)
    

    model = TimeSeriesModel(
        # ts=ts,
        freq_thresh=freq_thresh,
        lag=args.lag,
        type_=type_list,
        prcnt=args.prcnt,
        window=args.window,
        ema_span=args.ema_span,
        models=LinearRegression,
        metric=metric,
        models = models
    )

    results = model.run()
    print("Results:")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()