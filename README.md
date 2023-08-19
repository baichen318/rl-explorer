# Towards Automated RISC-V Microarchitecture Design with Reinforcement Learning

## Description

Microarchitecture determines the implementation of a microprocessor.
Designing a microarchitecture to achieve better performance, power, and area (PPA) trade-off has been increasingly difficult.
Previous data-driven methodologies hold inappropriate assumptions and failed to tightly couple with expert knowledge.
In this repo., we release a novel reinforcement learning-based (RL) solution that addresses these limitations.
With the integration of microarchitecture scaling graph, PPA preference space embedding, and proposed lightweight environment in RL, experiments using commercial electronic design automation (EDA) tools show that our method achieves an average PPA trade-off improvement of 16.03% than previous state-of-the-art approaches with 4.07× higher efficiency.
The solution qualities also outperform human implementations by at most 2.03× in the PPA trade-off.


## Folder organizations

```bash
```

## End-to-end Flow
- Environment setup
```bash
$ export PYTHONPATH=`pwd`
```

- Rocket@CUHK
```bash
$ python3 main.py -c configs/rocket.yml
```
- Rocket@CUHK
```bash
$ python3 main.py -c configs/boom-macos.yml
```

## Notice

This is a branch: that demonstrates the importance of the VLSI flow rather than the analysis tools.
