# Author: baichen318@gmail.com

import os

MACROS = {
    "root": os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        os.path.pardir,
        os.path.pardir
    ),
    "chipyard-root": "/research/dept8/gds/cbai/research/chipyard",
    "power-root": "/research/dept8/gds/cbai/research/synopsys-flow/build/pt-pwr"
}

def handle_macros():
    machine = os.popen("hostname").readlines()[0].strip()
    if machine == "cuhk":
        MACROS["gem5-root"] = os.path.join(
            os.sep, "home", "baichen", "cbai", "research", "gem5-repo"
        )
        MACROS["gem5-benchmark-root"] = os.path.join(
            os.sep, "home", "baichen", "cbai", "research", "data", "gem5-riscv-tests"
        )
    elif machine == "proj12":
        MACROS["gem5-root"] = os.path.join(
            os.sep, "home", "cbai", "research", "gem5-repo"
        )
        MACROS["gem5-benchmark-root"] = os.path.join(
            os.sep, "home", "cbai", "data", "gem5-riscv-tests"
        )
    elif machine.startswith("hpc"):
        MACROS["gem5-root"] = os.path.join(
            os.sep, "research", "dept8", "gds", "cbai", "research", "gem5-repo"
        )
        MACROS["gem5-benchmark-root"] = os.path.join(
            os.sep, "research", "dept8", "gds", "cbai", "data", "gem5-riscv-tests"
        )
    elif machine.startswith("dgg4"):
        pass
    else:
        print("[ERROR]: %s is not supported." % machine)
        exit(-1)

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

    MACROS["chipyard-vlsi-root"] = os.path.join(
        MACROS["chipyard-root"],
        "vlsi",
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
        "rocket-chip",
        "src",
        "main",
        "scala",
        "subsystem",
        "Configs.scala"
    )

    MACROS["rocket-configs"] = os.path.join(
        MACROS["chipyard-root"],
        "generators",
        "chipyard",
        "src",
        "main",
        "scala",
        "config",
        "RocketConfigs.scala"
    )

    MACROS["generate-auto-vlsi-v1"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "rocket",
        "scripts",
        "generate-auto-vlsi-v1.sh"
    )

    MACROS["generate-auto-vlsi-v2"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "rocket",
        "scripts",
        "generate-auto-vlsi-v2.sh"
    )

    MACROS["sim-script"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "rocket",
        "scripts",
        "sim.sh"
    )

    MACROS["run-script"] = os.path.join(
        MACROS["rl-explorer-root"],
        "vlsi",
        "rocket",
        "scripts",
        "run.tcl"
    )

    MACROS["tools-root"] = os.path.join(
        MACROS["root"],
        "tools"
    )

    MACROS["temp-root"] = os.path.join(
        MACROS["root"],
        "temp"
    )

    MACROS["mcpat-root"] = os.path.join(
        MACROS["tools-root"],
        "mcpat-riscv-7"
    )

handle_macros()
