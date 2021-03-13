# Author: baichen318@gmail.com

import os

MACROS = {
    "config_mixins": "/research/d3/cbai/research/chipyard/generators/boom/src/main/scala/common/config-mixins.scala",
    "boom_configs": "/research/d3/cbai/research/chipyard/generators/chipyard/src/main/scala/config/BoomConfigs.scala",
    "chipyard_root": "/research/d3/cbai/research/chipyard",
    "chipyard_vlsi_root": "/research/d3/cbai/research/chipyard/vlsi",
    "vlsi_root": "/research/d3/cbai/research/design-explorer/vlsi",
    "scripts": "/research/d3/cbai/research/design-explorer/vlsi/scripts",
    "power_root": "/uac/gds/cbai/cbai/research/synopsys-flow/build/pt-pwr/"
}

def modify_macros(core_name, soc_name):
    MACROS["compile-script"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "compile-%s.sh" % core_name
    )
    MACROS["simv-script"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "vcs-%s.sh" % core_name
    )
    MACROS["generated-src"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "generated-src",
        "chipyard.TestHarness.%s" % soc_name
    )
    MACROS["sim-syn-rundir"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "build",
        "chipyard.TestHarness.%s-ChipTop" % soc_name,
        "sim-syn-rundir"
    )
    MACROS["syn-rundir"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "build",
        "chipyard.TestHarness.%s-ChipTop" % soc_name,
        "syn-rundir"
    )
    MACROS["hir-file"] = os.path.join(
        MACROS["chipyard_root"],
        "chipyard.TestHarness.%s.hir" % soc_name
    )
    MACROS["sram-vhdl"] = os.path.join(
        MACROS["chipyard_vlsi_root"],
        "sram_behav_models.v"
    )
    MACROS["sim-path"] = os.path.join(
        MACROS["sim-syn-rundir"],
        "output"
    )
    MACROS["power-path"] = os.path.join(
        MACROS["power_root"],
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
