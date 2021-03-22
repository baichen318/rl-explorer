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
- ~~YAML, CSV cleaning~~
- ~~Report extraction verification~~
- ~~GP Initialization~~
- ~~GP models saving~~
- ~~Optimized point verification~~
- ~~Duplicated points verification~~
- VLSI error handler
- ~~Parallelization~~

## Unresolved Issues
Parallelization & Sync.

## Notice
- Pay attention to vlsi/macros.py
