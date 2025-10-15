# Time series smoothing\filtering methods comparing

## Usage


0. Download

```
git clone ...
```

1. Install python and requirements.
```
pip install -r requirements.txt
```

2. Train and check results. All experiment scripts in folder `/script`. Check results in `/experiments`.

```

#check models without smooth on ruo data
bash ./script/ruo_base.sh

#check models with ema smooth on sse50 data
bash ./script/sse_ema.sh

...
```
