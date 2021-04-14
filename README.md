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

<!-- ## Results
- LR@whetstone

![Linear Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/img/lr-whetstone.jpg "LR")

- Ridge@whetstone

![Ridge Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/img/ridge-whetstone.jpg "Ridge Regression")

- XGB@whetstone

![XGB Regression](https://gitee.com/baichen318/design-explorer/raw/dev-v4/data/img/xgb-whetstone.jpg "XGB Regression")
 -->

## TODO
<!-- - ~~Baseline design data@2GHz~~
- ~~Linear Regression~~
- ~~LASSO~~
- ~~Ridge~~
- ~~XGB~~
- ~~SVR~~
- ~~Linear SVR~~
- ~~Random Forest~~
- ~~AdaBoost~~
- ~~Bagging~~
- Offline Bayes + GP -->
- Two stages:
    * initialization
    * Searching - SA
- Initialization:
    * Configuration generation
    * Offline VLSI flow scripts
    * Automatic model training
    * Plotting
        * Sampling plotting
        * Prediction Plotting
        * VLSI verification plotting
        * Comparison between models & references
- Performance
    * Hyper Volume
    ```c
        HV = (abs(Latency_i - Latency_0) / Latency_0) * (abs(Power_i - Power_0) / Power_0)
        s.t. sign(Latency_i - Latency_0) + sign(Power_i - Power_0) = -2
    ```
