# Author: baichen318@gmail.com


import os
import re
import sys
import time
import numpy as np
import multiprocessing
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from simulation.base_simulation import Simulation
from utils.utils import execute, remove_prefix, if_exist, \
    remove, mkdir, round_power_of_two, error


class Gem5Wrapper(Simulation):
    """ Gem5Wrapper """
    def __init__(self, configs, design_space, state, idx):
        super(Gem5Wrapper, self).__init__(configs)
        self.design_space = design_space
        self.state = state
        self.idx = idx
        self.macros["gem5-research-root"] = os.path.abspath(
            os.path.join(
                self.configs["env"]["sim"]["gem5-research-root"],
                str(self.idx),
                "gem5-research"
            )
        )
        self.macros["gem5-benchmark-root"] = os.path.join(
            self.macros["gem5-research-root"],
            "benchmarks",
            "riscv-tests"
        )
        self.macros["simulator"] = "gem5-{}.opt".format(
            design_space.embedding_to_idx(state.tolist())
        )
        self.initialize_lut()
        self.stats, self.stats_name = self.init_stats()

    @property
    def benchmarks(self):
        return self.configs["env"]["benchmarks"]

    @property
    def logger(self):
        return self.configs["logger"]

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
        self.o3cpu_root = os.path.join(
            self.macros["gem5-research-root"],
            "src",
            "cpu",
            "o3",
            "O3CPU.py"
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


    def init_stats(self):
        # 16 stats
        stats = OrderedDict()
        stats_name = [
            "issueRate", "intInstQueueReads", "intInstQueueWrites",
            "intAluAccesses", "numLoadInsts", "numStoreInsts",
            "numBranches", "intRegfileReads", "intRegfileWrites"
        ]
        for name in stats_name:
            stats[name] = 0
        return stats, stats_name

    def modify_gem5(self):
        # NOTICE: we modify GEM5 accordingly
        def _modify_gem5(src, pattern, target, count=0):
            cnt = open(src, "r+").read()
            with open(src, 'w') as f:
                f.write(re.sub(r"%s" % pattern, target, cnt, count))

        # fetchWidth
        fetch_width = self.state[1]
        _modify_gem5(
            self.o3cpu_root,
            "fetchWidth\ =\ Param.Unsigned\(\d+,\ \"Fetch\ width\"\)",
            "fetchWidth = Param.Unsigned(%d, \"Fetch width\")" % (
                12 if (fetch_width << 1) > 12 else (fetch_width << 1)
            )
        )

        # decodeWidth
        decode_width = self.state[5]
        _modify_gem5(
            self.o3cpu_root,
            "decodeWidth\ =\ Param.Unsigned\(\d+,\ \"Decode\ width\"\)",
            "decodeWidth = Param.Unsigned(%d, \"Decode width\")" % decode_width
        )

        # numFetchBufferEntries
        # NOTICE: GEM5 requires `cache block size % fetch buffer == 0`
        fetch_buffer_entries = self.state[2]
        _modify_gem5(
            self.o3cpu_root,
            "fetchBufferSize\ =\ Param.Unsigned\(\d+,\ \"Fetch\ buffer\ size\ in\ bytes\"\)",
            "fetchBufferSize = Param.Unsigned(%d, \"Fetch buffer size in bytes\")" %
                int(round_power_of_two(fetch_buffer_entries))
        )

        # numRobEntries
        rob_entries = self.state[6]
        _modify_gem5(
            self.o3cpu_root,
            "numROBEntries\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ reorder\ buffer\ entries\"\)",
            "numROBEntries = Param.Unsigned(%d, \"Number of reorder buffer entries\")" %
                rob_entries
        )

        # numIntPhysRegisters
        int_phys_registers = self.state[7]
        _modify_gem5(
            self.o3cpu_root,
            "numPhysIntRegs\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ physical\ integer\ registers\"\)",
            "numPhysIntRegs = Param.Unsigned(%d, \"Number of physical integer registers\")" %
                int_phys_registers
        )

        # numFpPhysRegisters
        fp_phys_registers = self.state[8]
        _modify_gem5(
            self.o3cpu_root,
            "numPhysFloatRegs\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ physical\ floating\ point\ \"",
            "numPhysFloatRegs = Param.Unsigned(%d, \"Number of physical floating point \"" %
                fp_phys_registers
        )

        # numLdqEntries
        ldq_entries = self.state[18]
        _modify_gem5(
            self.o3cpu_root,
            "LQEntries\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ load\ queue\ entries\"\)",
            "LQEntries = Param.Unsigned(%d, \"Number of load queue entries\")" %
                ldq_entries
        )

        # numStqEntries
        stq_entries = self.state[19]
        _modify_gem5(
            self.o3cpu_root,
            "SQEntries\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ store\ queue\ entries\"\)",
            "SQEntries = Param.Unsigned(%d, \"Number of store queue entries\")" %
                stq_entries
        )

        # issueEntries
        # NOTICE: GEM5 requires `issueWidth <= 12`
        issue_width = self.state[9] + self.state[12] + self.state[15]
        _modify_gem5(
            self.o3cpu_root,
            "issueWidth\ =\ Param.Unsigned\(\d+,\ \"Issue\ width\"\)",
            "issueWidth = Param.Unsigned(%d, \"Issue width\")" % (
                12 if issue_width > 12 else issue_width
            )
        )

        # issueQueue
        issue_queue = self.state[10] + self.state[13] + self.state[16]
        _modify_gem5(
            self.o3cpu_root,
            "numIQEntries\ =\ Param.Unsigned\(\d+,\ \"Number\ of\ instruction\ queue\ entries\"\)",
            "numIQEntries = Param.Unsigned(%d, \"Number of instruction queue entries\")" % (
                issue_queue
            )
        )

        # dcache.nMSHRs
        mshrs = self.state[24]
        _modify_gem5(
            self.cache_root,
            "mshrs\ =\ \d+",
            "mshrs = %d" % round_power_of_two(mshrs),
            count=1
        )

        # DEPRECATED: dcache.nTLBWays
        # dcache_tlb_ways = self.state[22]
        # _modify_gem5(
        #     self.tlb_root,
        #     "size\ =\ Param\.Int\(\d+,\ \"TLB\ size\"\)",
        #     "size = Param.Int(%d, \"TLB size\")" %
        #         round_power_of_two(dcache_tlb_ways)
        # )

        # BTB
        """
            common/parameters.scala:
                numRasEntries: Int = 32
        """
        _modify_gem5(
            self.btb_root,
            "RASSize\ =\ Param\.Unsigned\(\d+,\ \"RAS\ size\"\)",
            "RASSize = Param.Unsigned(32, \"RAS size\")"
        )

        # BTBEntries
        """
            ifu/bpd/btb.scala:
                case class BoomBTBParams(
                  nSets: Int = 128,
                  nWays: Int = 2,
                  offsetSz: Int = 13,
                  extendedNSets: Int = 128
                )
        """
        _modify_gem5(
            self.btb_root,
            "BTBEntries\ =\ Param\.Unsigned\(\d+,\ \"Number\ of\ BTB\ entries\"\)",
            "BTBEntries = Param.Unsigned(512, \"Number of BTB entries\")"
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
        misc_stats = OrderedDict()
        with open(os.path.join(self.m5out_root, "stats.txt"), 'r') as f:
            cnt = f.readlines()
        stats_name = ["system.cpu.{}".format(name) for name in self.stats_name]
        for line in cnt:
            if line.startswith("simInsts"):
                instructions = int(line.split()[1])
            if line.startswith("system.cpu.numCycles"):
                cycles = int(line.split()[1])
            for name in stats_name:
                if line.startswith(name):
                    misc_stats[remove_prefix(name, "system.cpu.")] = float(line.split()[1])
        return instructions, cycles, misc_stats

    def incr_stats(self, misc_stats):
        for k, v in misc_stats.items():
            self.stats[k] += v

    def avg_stats(self, cnt=1):
        for k, v in self.stats.items():
            self.stats[k] /= cnt

    def simulate(self):
        machine = os.popen("hostname").readlines()[0].strip()
        for bmark in self.benchmarks:
            remove(os.path.join(
                self.temp_root, "m5out-{}".format(bmark))
            )
        ipc = 0
        bp = {
            1: "LTAGE",
            2: "TAGE",
            3: "BiModeBP"
        }
        for bmark in self.benchmarks:
            cmd = "cd {}; build/RISCV/{} configs/example/se.py ".format(
                self.macros["gem5-research-root"],
                self.macros["simulator"]
            )
            cmd += "--cmd=%s " % os.path.join(
                self.macros["gem5-benchmark-root"],
                bmark + ".riscv"
            )
            cmd += "--num-cpus=1 "
            cmd += "--cpu-type=DerivO3CPU "
            cmd += "--caches "
            cmd += "--cacheline_size=64 "
            cmd += " --l1d_size={}kB ".format(
                (((self.state[22] * self.state[23]) << 6)) >> 10
            )
            cmd += "--l1i_size={}kB ".format(
                (((self.state[20] * self.state[21]) << 6)) >> 10
            )
            cmd += "--l1d_assoc={} ".format(
                self.state[22]
            )
            cmd += "--l1i_assoc={} ".format(
                self.state[20]
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
            cmd += "--bp-type=%s; cd -" % (
                bp[self.state[0]]
            )
            execute(cmd, logger=self.logger)
            instructions, cycles, misc_stats = self.get_results()
            self.incr_stats(misc_stats)
            ipc += (instructions / cycles)
            # for McPAT usage
            execute(
                "mv -f %s %s" % (
                    self.m5out_root,
                    os.path.join(self.temp_root, "m5out-%s" % bmark)
                )
            )
        ipc /= len(self.benchmarks)
        self.avg_stats(len(self.benchmarks))
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
            return ipc, self.stats
        self.modify_gem5()
        self.generate_gem5()
        ipc = self.simulate()
        return ipc, self.stats

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
        pool = ThreadPool(len(self.benchmarks))
        for bmark in self.benchmarks:
            mcpat_xml = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.xml" % ("BOOM", self.idx)
            )
            mcpat_report = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.rpt" % ("BOOM", self.idx)
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
                            "boom.xml"
                        ),
                        ' '.join([str(s) for s in self.state]),
                        mcpat_xml,
                        os.path.join(
                            self.macros["rl-explorer-root"],
                            "tools",
                            "mcpat-research",
                            "mcpat"
                        ),
                        mcpat_xml,
                        mcpat_report
                    ),
                )
            )
        pool.close()
        pool.join()
        for bmark in self.benchmarks:
            mcpat_report = os.path.join(
                self.temp_root,
                "m5out-%s" % bmark,
                "%s-%s.rpt" % ("BOOM", self.idx)
            )
            power += extract_power(mcpat_report)
            area += extract_area(mcpat_report)
        return power / len(self.benchmarks), \
            area / len(self.benchmarks)
