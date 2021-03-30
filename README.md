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

## TODO
- Baseline design data@2GHz
- Offline Bayes + GP
- Linear Regression
- LASSO
- Ridge
- ElasticNet
- BayesRidge
- SVR
