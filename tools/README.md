# Calibration

## Usage

- Get started with the help menu
```bash
$ python3 calib.py -h
```

- Generate the simulation data
```bash
$ python3 calib.py -c ../configs/boom-macos.yml -m simulation
```

- Calibration
```bash
$ python3 calib.py -c ../configs/boom-macos.yml -m calib
```

## Statistics

- Before calibration

SonicBOOM:

1. performance Kentall's tau: 0.6176692725770059

2. power Kentall's tau: 0.576524849754375

3. area Kentall's tau: 0.7601817447951914

- After calibration

SonicBOOM:

1. performance Kentall's tau: 0.8544

2. power Kentall's tau: 0.8670

3. area Kentall's tau: 0.9341
