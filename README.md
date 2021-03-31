# design-explorer
Design Explorer: Design Space Explorer focusing on the CPU design

## Data preparation
```bash
$ bash exp/exp-data.sh
```

## Visualize the Pareto frontier
```bash
$ bash exp/exp-vis.sh
```

## Train XGB Regressor
```bash
$ bash exp/exp-model.sh
```

## Offline dataset sampling
```bash
$ bash exp/exp-sample.sh
```

## Bayes Optimization Flow
```bash
$ bash exp/exp-opt.sh
```

## Results
- LR@whetstone

![Linear Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/lr-whetstone.jpg "LR")

- Ridge@whetstone

![Ridge Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/ridge-whetstone.jpg "Ridge Regression")

- XGB@whetstone

![XGB Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/xgb-whetstone.jpg "XGB Regression")

## TODO
- ~~Baseline design data@2GHz~~
- Offline Bayes + GP
- ~~Linear Regression~~
- ~~LASSO~~
- ~~Ridge~~
- ~~XGB~~
- ~~SVR~~
- ~~Linear SVR~~
- ~~Random Forest~~
- ~~AdaBoost~~
- ~~Bagging~~

