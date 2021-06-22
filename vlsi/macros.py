# Author: baichen318@gmail.com

import os

MACROS = {
    "chipyard-root": "/research/dept8/gds/cbai/research/chipyard",
    "rl-explorer-root": "/research/dept8/gds/cbai/research/rl-explorer",
    "benchmarks": [
        "median",
        "qsort",
        "rsort",
        "towers",
        "vvadd",
        "multiply",
        "dhrystone",
        "spmv",
        "mt-vvadd",
        "mt-matmul"
    ]
}

def handle_macros():
    MACROS["chipyard-sims-root"] = os.path.join(
        MACROS["chipyard-root"],
        "sims",
        "vcs"
    )

    MACROS["config-mixins"] = os.path.join(
        MACROS["chipyard-root"],
        "generators",
        "boom",
        "src",
        "main",
        "scala",
        "common",
        "config-mixins.scala"
    )

    MACROS["boom-configs"] = os.path.join(
        MACROS["chipyard-root"],
        "generators",
        "chipyard",
        "src",
        "main",
        "scala",
        "config",
        "BoomConfigs.scala"
    )

    MACROS["sim.sh"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "scripts",
        "sim.sh"
    )

handle_macros()
