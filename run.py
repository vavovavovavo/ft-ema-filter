from source.TS_model import *
import warnings
from openpyxl.styles.stylesheet import warn
import argparse
import json
import os

# Подавляем конкретное предупреждение
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TS_model')
    
    #basic
    parser.add_argument('--ts_name', type=str, required=True, default='RUO')
    parser.add_argument('--type', type=str, required=True, default='BASE')
    parser.add_argument('--lag', type=int, required=True, default=7)
    parser.add_argument('--prcnt', type=float, required=True, default=0.85)
    parser.add_argument('--metric', type=str, required=True, default='MAE_MAPE')
    parser.add_argument('--model_name', type=str, required=True, default='Ridge')
    

    #smooth params
    parser.add_argument('--freq_thresh', type=float, required=True, default=17)
    parser.add_argument('--window', type=int, required=True, default=40)
    parser.add_argument('--ema_span', type=int, required=True, default=5)
    
    
    #Ridge
    parser.add_argument('--alpha', type=float, default=0.01)

    #Random Forest
    parser.add_argument('--criterion', type=str, default='absolute_error')
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--n_estimators', type=int, default=400)

    #XGB
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
        
    args = parser.parse_args()
    print("=== Параметры запуска ===")
    for key, value in vars(args).items():
        print(f"{key:<20} : {value}")
        
    model = TS_model(
        **vars(args)
    )

    model.create_data()
    res = model.run()
    
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.ts_name}_{args.type}.json")
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results[args.model_name] = res

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Результат для {args.model_name} сохранён в {output_file}")


    # rf_best_ruo = {
    # 'criterion' : 'absolute_error',
    # 'max_depth' : 10,
    # 'min_samples_leaf' : 2, 
    # 'n_estimators' : 400
    # }

    # xgb_best_ruo = {
    # 'gamma' : 0, 
    # 'learning_rate' : 0.1, 
    # 'max_depth' : 3,
    # 'n_estimators' : 100,
    # }


    # model = TS_model(
    #     ts = 'RUO',
    #     type = 'KALMAN',
    #     window=40,
    #     freq_thresh=17,
    #     **xgb_best_ruo,
    #     model_name='XGB'
    # )
    # model.create_data()
    # res = model.run()
    # print(res)