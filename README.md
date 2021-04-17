# design-explorer
Design Explorer: Design Space Explorer focusing on the CPU design

## End-to-end Flow
```bash
$ python main.py -c configs/design-explorer.yml
```

## TODO
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

