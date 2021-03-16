# Author: baichen318@gmail.com

import os

MACROS = {
    "config-mixins": "/research/dept8/gds/cbai/research/chipyard/generators/boom/src/main/scala/common/config-mixins.scala",
    "boom-configs": "/research/dept8/gds/cbai/research/chipyard/generators/chipyard/src/main/scala/config/BoomConfigs.scala",
    "chipyard-root": "/research/dept8/gds/cbai/research/chipyard",
    "chipyard-vlsi-root": "/research/dept8/gds/cbai/research/chipyard/vlsi",
    "vlsi-root": "/research/dept8/gds/cbai/research/design-explorer/vlsi",
    "scripts": "/research/dept8/gds/cbai/research/design-explorer/vlsi/scripts",
    "power-root": "/research/dept8/gds/cbai/research/synopsys-flow/build/pt-pwr",
    "temp-sim-root": "/research/dept8/gds/cbai/temp"
}

def modify_macros(core_name, soc_name):
    MACROS["compile-script"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "compile-%s.sh" % core_name
    )
    MACROS["simv-script"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "vcs-%s.sh" % core_name
    )
    MACROS["generated-src"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "generated-src",
        "chipyard.TestHarness.%s" % soc_name
    )
    MACROS["sim-syn-rundir"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "build",
        "chipyard.TestHarness.%s-ChipTop" % soc_name,
        "sim-syn-rundir"
    )
    MACROS["syn-rundir"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "build",
        "chipyard.TestHarness.%s-ChipTop" % soc_name,
        "syn-rundir"
    )
    MACROS["hir-file"] = os.path.join(
        MACROS["chipyard-root"],
        "chipyard.TestHarness.%s.hir" % soc_name
    )
    MACROS["sram-vhdl"] = os.path.join(
        MACROS["chipyard-vlsi-root"],
        "sram_behav_models.v"
    )
    MACROS["sim-path"] = os.path.join(
        MACROS["sim-syn-rundir"],
        "output"
    )
    MACROS["temp-sim-path"] = os.path.join(
        MACROS["temp-sim-root"],
        "%s" % core_name
    )
    MACROS["power-path"] = os.path.join(
        MACROS["power-root"],
        "%s-benchmarks" % core_name
    )
    MACROS["temp-latency-yml"] = os.path.join(
        os.path.abspath(os.curdir),
        "configs",
        "%s-latency.yml" % core_name
    )
    MACROS["temp-power-yml"] = os.path.join(
        os.path.abspath(os.curdir),
        "configs",
        "%s-power.yml" % core_name
    )
    MACROS["temp-area-yml"] = os.path.join(
        os.path.abspath(os.curdir),
        "configs",
        "%s-area.yml" % core_name
    )
    MACROS["temp-latency-csv"] = os.path.join(
        os.path.abspath(os.curdir),
        "data",
        "%s-latency.csv" % core_name
    )
    MACROS["temp-power-csv"] = os.path.join(
        os.path.abspath(os.curdir),
        "data",
        "%s-power.csv" % core_name
    )
    MACROS["temp-area-csv"] = os.path.join(
        os.path.abspath(os.curdir),
        "data",
        "%s-area.csv" % core_name
    )
