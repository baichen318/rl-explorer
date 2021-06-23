# Author: baichen318@gmail.com

import os

MACROS = {
    "chipyard-root": "/research/dept8/gds/cbai/research/chipyard",
    "rl-explorer-root": "/research/dept8/gds/cbai/research/rl-explorer"
}

def handle_macros():
    MACROS["chipyard-sims-root"] = os.path.join(
        MACROS["chipyard-root"],
        "sims",
        "vcs"
    )

    MACROS["chipyard-sims-output-root"] = os.path.join(
        MACROS["chipyard-root"],
        "sims",
        "vcs",
        "output"
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

    MACROS["compile-script"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "scripts",
        "compile.sh"
    )

    MACROS["sim-script"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "scripts",
        "sim.sh"
    )

handle_macros()
