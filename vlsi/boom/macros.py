# Author: baichen318@gmail.com

import os

MACROS = {
    "chipyard-root": "/research/dept8/gds/cbai/research/chipyard",
}

def handle_macros():
    MACROS["rl-explorer-root"] = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.path.pardir,
        os.path.pardir
    )

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

    MACROS["generate-auto-vlsi"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "scripts",
        "generate-auto-vlsi.sh"
    )

    MACROS["sim-script"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "scripts",
        "sim.sh"
    )

handle_macros()