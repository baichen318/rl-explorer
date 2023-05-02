# Author: baichen318@gmail.com


import os
import re
import sys
import time
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool
from simulation.base_simulation import Simulation
from utils.utils import execute, if_exist, remove, mkdir, \
    round_power_of_two, error


class Gem5Wrapper(Simulation):
    """ Gem5Wrapper """
    def __init__(self, configs, design_space, state, idx):
        super(Gem5Wrapper, self).__init__(configs)
        self.design_space = design_space
        self.state = state
        self.idx = idx
        self.macros["gem5-research-root"] = os.path.abspath(
            os.path.join(
                self.configs["gem5-research-root"],
                "gem5-research"
            ),
        )
        self.macros["gem5-benchmark-root"] = os.path.join(
            self.macros["gem5-research-root"],
            "benchmarks",
            "riscv-tests"
        )
        self.macros["simulator"] = "gem5-{}-{}.opt".format(
            self.state[0],
            self.state[5]
        )
        self.initialize_lut()

    def initialize_lut(self):
        self.btb_root = os.path.join(
            self.macros["gem5-research-root"],
            "src",
            "cpu",
            "pred",
            "BranchPredictor.py"
        )
        self.tlb_root = os.path.join(
            self.macros["gem5-research-root"],
            "src",
            "arch",
            "riscv",
            "RiscvTLB.py"
        )
        self.cache_root = os.path.join(
            self.macros["gem5-research-root"],
            "configs",
            "common",
            "Caches.py"
        )
        self.m5out_root = os.path.join(
            self.macros["gem5-research-root"],
            "m5out"
        )
        self.temp_root = os.path.join(
            self.macros["rl-explorer-root"],
            "temp",
            str(self.idx)
        )
        mkdir(self.temp_root)


    def modify_gem5(self):
        # NOTICE: we modify gem5 w.r.t. state[0] & state[5]
        def _modify_gem5(src, pattern, target, count=0):
            cnt = open(src, "r+").read()
            with open(src, 'w') as f:
                f.write(re.sub(r"%s" % pattern, target, cnt, count))

        # RAS@btb
        ras_size = self.design_space.get_mapping_params(self.state, 0)[0]
        _modify_gem5(
            self.btb_root,
            "RASSize\ =\ Param\.Unsigned\(\d+,\ \"RAS\ size\"\)",
            "RASSize = Param.Unsigned(%d, \"RAS size\")" % (
                4 if ras_size == 0 else \
                    round_power_of_two(ras_size)
            )
        )

        # BTB@btb
        btb = self.design_space.get_mapping_params(self.state, 0)[1]
        _modify_gem5(
            self.btb_root,
            "BTBEntries\ =\ Param\.Unsigned\(\d+,\ \"Number\ of\ BTB\ entries\"\)",
            "BTBEntries = Param.Unsigned(%d, \"Number of BTB entries\")" % (
                2 if btb == 0 else \
                    round_power_of_two(btb)
            )
        )

        # TLB@D-Cache
        tlb = self.design_space.get_mapping_params(self.state, 5)[2]
        _modify_gem5(
            self.tlb_root,
            "size\ =\ Param\.Int\(\d+,\ \"TLB\ size\"\)",
            "size = Param.Int(%d, \"TLB size\")" % (
                2 if tlb == 0 else \
                    round_power_of_two(tlb)
            )
        )

        # MSHR@D-Cache
        mshr = self.design_space.get_mapping_params(self.state, 5)[3]
        _modify_gem5(
            self.cache_root,
            "mshrs\ =\ \d+",
            "mshrs = %d" % (
                1 if mshr == 0 else \
                    round_power_of_two(mshr)
            ),
            count=1
        )

    def generate_gem5(self):
        # NOTICE: commands are manually designed
        machine = os.popen("hostname").readlines()[0].strip()
        if machine == "cuhk":
            cmd = "cd %s; " % self.macros["gem5-research-root"]
            cmd += "/home/baichen/cbai/tools/Python-3.9.7/build/bin/scons "
            cmd += "build/RISCV/gem5.opt PYTHON_CONFIG=\"/home/baichen/cbai/tools/Python-3.9.7/build/bin/python3-config\" "
            cmd += "-j%d; " % int(round(1.4 * multiprocessing.cpu_count()))
            cmd += "mv build/RISCV/gem5.opt build/RISCV/{}; ".format(
                self.macros["simulator"]
            )
            cmd += "cd -;"
        elif machine == "proj12":
            cmd = "cd %s; " % self.macros["gem5-research-root"]
            cmd += "scons "
            cmd += "build/RISCV/gem5.opt "
            cmd += "-j%d; " % int(round(2 * multiprocessing.cpu_count()))
            cmd += "mv build/RISCV/gem5.opt build/RISCV/{}; ".format(
                self.macros["simulator"]
            )
            cmd += "cd -;"
        elif machine.startswith("hpc"):
            cmd = "cd %s; " % self.macros["gem5-research-root"]
            cmd += "/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/scons "
            cmd += "build/RISCV/gem5.opt CCFLAGS_EXTRA=\"-I/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/include\" "
            cmd += "PYTHON_CONFIG=\"/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/python3-config\" "
            cmd += "LDFLAGS_EXTRA=\"-L/research/dept8/gds/cbai/tools/protobuf-3.6.1/build/lib -L/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/lib\" "
            cmd += "-j%d; " % int(round(1.4 * multiprocessing.cpu_count()))
            cmd += "mv build/RISCV/gem5.opt build/RISCV/{}; ".format(
                self.macros["simulator"]
            )
            cmd += "cd -;"
        elif machine.startswith("alish-rs"):
            cmd = "cd {} && " \
                "scons build/RISCV/gem5.opt CCFLAGS_EXTRA=\"-I/proj/users/chen.bai/tools/protobuf-21.6/build/include\" " \
                "PYTHON_CONFIG=/usr/bin/python3-config " \
                "LINKFLAGS_EXTRA=\"-L/proj/users/chen.bai/tools/protobuf-21.6/build/lib\" " \
                "-j{}; ".format(
                    self.macros["gem5-research-root"],
                    int(multiprocessing.cpu_count())
                )
            cmd += "mv build/RISCV/gem5.opt build/RISCV/{}; ".format(
                self.macros["simulator"]
            )
        elif "MacBook-Pro" in machine or "192.168" in machine:
            cmd = "cd %s; " % self.macros["gem5-research-root"]
            cmd += "scons "
            cmd += "build/RISCV/gem5.opt "
            cmd += "-j%d; " % int(round(multiprocessing.cpu_count()))
            cmd += "mv build/RISCV/gem5.opt build/RISCV/{}; ".format(
                self.macros["simulator"]
            )            
            cmd += "cd -;"
        else:
            error("{} is not support.".format(machine))
            exit(-1)
        execute(cmd)

    def get_results(self):
        instructions, cycles = 0, 0
        with open(os.path.join(self.m5out_root, "stats.txt"), 'r') as f:
            cnt = f.readlines()
        for line in cnt:
            if line.startswith("simInsts"):
                instructions = int(line.split()[1])
            if line.startswith("system.cpu.numCycles"):
                cycles = int(line.split()[1])
        return instructions, cycles

    def simulate(self):
        machine = os.popen("hostname").readlines()[0].strip()
        for bmark in self.configs["benchmarks"]:
            remove(os.path.join(self.temp_root, "m5out-%s" % bmark))
        ipc = 0
        for bmark in self.configs["benchmarks"]:
            cmd = "cd {}; build/RISCV/{} configs/example/se.py ".format(
                self.macros["gem5-research-root"],
                self.macros["simulator"]
            )
            cmd += "--cmd=%s " % os.path.join(
                self.macros["gem5-benchmark-root"],
                bmark + ".riscv"
            )
            cmd += "--num-cpus=1 "
            cmd += "--cpu-type=TimingSimpleCPU "
            cmd += "--caches "
            cmd += "--cacheline_size=64 "
            cmd += " --l1d_size={}kB ".format(
                (
                    self.design_space.get_mapping_params(self.state, 5)[0] * \
                        self.design_space.get_mapping_params(self.state, 5)[1] * \
                            (2 ** 6)
                ) >> 10
            )
            cmd += "--l1i_size={}kB ".format(
                (
                    self.design_space.get_mapping_params(self.state, 1)[0] * \
                        (2 ** 6) * \
                            (2 ** 6)
                ) >> 10
            )
            cmd += "--l1d_assoc={} ".format(
                self.design_space.get_mapping_params(self.state, 5)[1]
            )
            cmd += "--l1i_assoc={} ".format(
                self.design_space.get_mapping_params(self.state, 1)[0]
            )
            cmd += "--sys-clock=2000000000Hz "
            cmd += "--cpu-clock=2000000000Hz "
            cmd += "--sys-voltage=6.3V "
            # cmd += "--l2cache "
            # cmd += "--l2_size=64MB "
            # cmd += "--l2_assoc=8 "
            cmd += "--mem-size=4096MB "
            cmd += "--mem-type=LPDDR3_1600_1x32 "
            cmd += "--mem-channels=1 "
            cmd += "--enable-dram-powerdown "
            cmd += "--bp-type=BiModeBP "
            cmd += "--l1i-hwp-type=TaggedPrefetcher "
            cmd += "--l1d-hwp-type=TaggedPrefetcher "
            cmd += "--l2-hwp-type=TaggedPrefetcher; cd -"
            execute(cmd, logger=self.configs["logger"])
            instructions, cycles = self.get_results()
            ipc += (instructions / cycles)
            # for McPAT usage
            execute(
                "mv -f %s %s" % (
                    self.m5out_root,
                    os.path.join(self.temp_root, "m5out-%s" % bmark)
                )
            )
        ipc /= len(self.configs["benchmarks"])
        return ipc


    def evaluate_perf(self):
        if if_exist(
            os.path.join(
                self.macros["gem5-research-root"],
                "build",
                "RISCV",
                "{}".format(
                    self.macros["simulator"]
                )
            )
        ):
            ipc = self.simulate()
            return ipc
        self.modify_gem5()
        self.generate_gem5()
        ipc = self.simulate()
        return ipc

    def evaluate_power_and_area(self):
        def extract_power(mcpat_report):
            # p_subthreshold = re.compile(r"Subthreshold\ Leakage\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            # p_gate = re.compile(r"Gate\ Leakage\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            p_dynamic = re.compile(r"Runtime\ Dynamic\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            subthreshold, gate, dynamic = 0, 0, 0
            with open(mcpat_report, 'r') as rpt:
                rpt = rpt.read()
                try:
                    # subthreshold = float(p_subthreshold.findall(rpt)[1][0])
                    # gate = float(p_gate.findall(rpt)[1][0])
                    dynamic = float(p_dynamic.findall(rpt)[1][0])
                except Exception as e:
                    error(str(e))
                    exit(1)
            return subthreshold + gate + dynamic

        def extract_area(mcpat_report):
            p_area = re.compile(r"Area\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ mm\^2")
            area = 0
            with open(mcpat_report, 'r') as rpt:
                rpt = rpt.read()
                try:
                    area = float(p_area.findall(rpt)[1][0])
                except Exception as e:
                    error(str(e))
                    exit(1)
            return area

        power, area = 0, 0
        pool = ThreadPool(len(self.configs["benchmarks"]))
        for bmark in self.configs["benchmarks"]:
            mcpat_xml = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.xml" % ("Rocket", self.idx)
            )
            mcpat_report = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.rpt" % ("Rocket", self.idx)
            )
            pool.apply_async(
                execute,
                (
                    "{} {} " \
                    "-y {} " \
                    "-c {} " \
                    "-s {} " \
                    "-t {} " \
                    "--state {} " \
                    "-o {}; " \
                    "{} " \
                    "-infile {} " \
                    "-print_level 5 > {}" \
                    .format(
                        sys.executable, os.path.join(
                            self.macros["rl-explorer-root"],
                            "tools",
                            "gem5-mcpat-parser.py"
                        ),
                        self.configs["configs"],
                        os.path.join(self.temp_root, "m5out-{}".format(bmark), "config.json"),
                        os.path.join(self.temp_root, "m5out-{}".format(bmark), "stats.txt"),
                        os.path.join(
                            self.macros["rl-explorer-root"],
                            "tools",
                            "template",
                            "rocket.xml"
                        ),
                        ' '.join([str(s) for s in self.state]),
                        mcpat_xml,
                        os.path.join(
                            self.macros["rl-explorer-root"],
                            "tools",
                            "mcpat-riscv-7",
                            "mcpat"
                        ),
                        mcpat_xml,
                        mcpat_report
                    ),
                )
            )
        pool.close()
        pool.join()
        for bmark in self.configs["benchmarks"]:
            mcpat_report = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.rpt" % ("Rocket", self.idx)
            )
            power += extract_power(mcpat_report)
            area += extract_area(mcpat_report)
        return power / len(self.configs["benchmarks"]), \
            area / len(self.configs["benchmarks"])
