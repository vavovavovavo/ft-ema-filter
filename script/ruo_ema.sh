ts_name='RUO'


python3 -u run.py \
    --ts_name $ts_name \
    --model_name 'Ridge' \
    --type 'EMA' \
    --lag 7 \
    --prcnt 0.85 \
    --metric 'MAE_MAPE' \
    --freq_thresh 17 \
    --window  40 \
    --ema_span 5 \
    --alpha 0.01 

python3 -u run.py \
    --ts_name $ts_name \
    --model_name 'Random Forest' \
    --type 'EMA' \
    --lag 7 \
    --prcnt 0.85 \
    --metric 'MAE_MAPE' \
    --freq_thresh 17 \
    --window  40 \
    --ema_span 5 \
    --criterion 'absolute_error' \
    --max_depth 10 \
    --min_samples_leaf 2 \
    --n_estimators 400


python3 -u run.py \
    --ts_name $ts_name \
    --model_name 'XGB' \
    --type 'EMA' \
    --lag 7 \
    --prcnt 0.85 \
    --metric 'MAE_MAPE' \
    --freq_thresh 17 \
    --window  40 \
    --ema_span 5 \
    --gamma 0.0 \
    --learning_rate 0.1 \
    --max_depth 3 \
    --n_estimators 100

