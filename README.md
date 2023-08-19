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
rl-explorer
├── LICENSE
├── README.md
├── __init__.py
├── baselines
│	 ├── boom_explorer
│	 │	 ├── LICENSE
│	 │	 ├── README.md
│	 │	 ├── algo
│	 │	 │	 ├── boom_explorer.py
│	 │	 │	 ├── dkl_gp.py
│	 │	 │	 └── problem.py
│	 │	 ├── configs
│	 │	 │	 ├── README.md
│	 │	 │	 └── boom-explorer.yml
│	 │	 ├── main.py
│	 │	 └── util
│	 │	     ├── __init__.py
│	 │	     ├── exception.py
│	 │	     ├── sample.py
│	 │	     └── util.py
│	 ├── dac16
│	 │	 ├── README.md
│	 │	 └── dac16.py
│	 └── isca14
│	     ├── README.md
│	     ├── boom.txt
│	     ├── configs
│	     │	 └── isca14.yml
│	     ├── isca14.py
├── data
│	 ├── boom
│	 │	 ├── README.md
│	 │	 ├── boom.txt
│	 │	 └── dataset.txt
│	 └── rocket
│	     ├── dataset.txt
│	     └── rocket.txt
├── dse
│	 ├── algo
│	 │	 └── a3c
│	 │	     ├── a3c.py
│	 │	     ├── agent
│	 │	     │	 ├── agent.py
│	 │	     │	 ├── boom.py
│	 │	     │	 └── rocket.py
│	 │	     ├── buffer.py
│	 │	     ├── functions.py
│	 │	     ├── model.py
│	 │	     └── preference.py
│	 └── env
│	     ├── base_design_space.py
│	     ├── boom
│	     │	 ├── design_space.py
│	     │	 └── env.py
│	     └── rocket
│	         ├── design_space.py
│	         └── env.py
├── main
│	 ├── configs
│	 │	 ├── example.yml
│	 │	 ├── giga.yaml
│	 │	 ├── medium.yaml
│	 │	 ├── mega.yaml
│	 │	 ├── rocket.yaml
│	 │	 └── small.yaml
│	 └── main.py
├── simulation
│	 ├── base_simulation.py
│	 ├── boom
│	 │	 └── simulation.py
│	 └── rocket
│	     └── simulation.py
├── tools
│	 ├── README.md
│	 ├── calib.py
│	 ├── gem5-mcpat-parser.py
│	 ├── mcpat-research
│	 │	 └── ...                        # mcpat-research project
│	 ├── models
│	 │	 ├── boom
│	 │	 │	 ├── boom-area.pt
│	 │	 │	 ├── boom-perf.pt
│	 │	 │	 ├── boom-power.pt
│	 │	 └── rocket
│	 │	     ├── rocket-area.pt
│	 │	     ├── rocket-perf.pt
│	 │	     └── rocket-power.pt
│	 └── template
│	  	 ├── boom.xml
│	  	 ├── rocket.xml
│	  	 └── template.xml
└── utils
    ├── exceptions.py
    ├── handle_data.py
    ├── thread.py
    ├── utils.py
    └── visualizer.py
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
