# Author: baichen318@gmail.com


import os
import re
import sys
import time
import numpy as np
import multiprocessing
from collections import OrderedDict
from utils.thread import WorkerThread
from multiprocessing.pool import ThreadPool
from simulation.base_simulation import Simulation
from utils.utils import execute, remove_prefix, if_exist, \
    remove, mkdir, round_power_of_two, error, assert_error


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
        if isinstance(state, np.ndarray):
            state = state.tolist()
        self.macros["simulator"] = "gem5-{}.opt".format(
            design_space.embedding_to_idx(state)
        )
        self.initialize_lut()
        self.stats, self.stats_name = self.init_stats()
        self.simulation_is_failed = False

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
        self.m5out_root = os.path.join(
            self.macros["gem5-research-root"],
            "m5out"
        )
        """
            DEPRECATED.
            `temp_root` is deprecated for multiple
            simultaneous simulations.
        """
        # self.temp_root = os.path.join(
        #     self.macros["towards-automated-risc-v-microarchitecture-design-with-reinforcement-learning-root"],
        #     "temp",
        #     str(self.idx)
        # )
        # mkdir(self.temp_root)

    def init_stats(self):
        # 9 stats
        stats = OrderedDict()
        if "BOOM" in self.configs["algo"]["design"]:
            stats_name = [
                "issueRate", "intInstQueueReads", "intInstQueueWrites",
                "intAluAccesses", "numLoadInsts", "numStoreInsts",
                "numBranches", "intRegfileReads", "intRegfileWrites"
            ]
        else:
            assert "Rocket" in self.configs["algo"]["design"], \
                assert_error("{} is unsupported.".format(
                    configs["algo"]["design"])
                )
            stats_name = [
                "branchPred.condIncorrect", "branchPred.RASIncorrect", "system.cpu.dcache.ReadReq.mshrMissRate::total",
                "system.cpu.dcache.WriteReq.mshrMissRate::total", "exec_context.thread_0.numInsts", "exec_context.thread_0.numIntAluAccesses",
                "exec_context.thread_0.numLoadInsts", "exec_context.thread_0.numStoreInsts", "exec_context.thread_0.numBranches"
            ]
        for name in stats_name:
            stats[name] = 0
        return stats, stats_name

    def modify_gem5(self):
        # NOTICE: we modify gem5 w.r.t. state[0] & state[5]
        def _modify_gem5(src, pattern, target, count=0):
            cnt = open(src, "r+").read()
            with open(src, 'w') as f:
                f.write(re.sub(r"%s" % pattern, target, cnt, count))

        # RAS@btb
        ras_size = self.state[0]
        _modify_gem5(
            self.btb_root,
            "RASSize\ =\ Param\.Unsigned\(\d+,\ \"RAS\ size\"\)",
            "RASSize = Param.Unsigned(%d, \"RAS size\")" % (
                4 if ras_size == 0 else \
                    round_power_of_two(ras_size)
            )
        )

        # BTB@btb
        btb = self.state[1]
        _modify_gem5(
            self.btb_root,
            "BTBEntries\ =\ Param\.Unsigned\(\d+,\ \"Number\ of\ BTB\ entries\"\)",
            "BTBEntries = Param.Unsigned(%d, \"Number of BTB entries\")" % (
                2 if btb == 0 else \
                    round_power_of_two(btb)
            )
        )

        # TLB@D-Cache
        tlb = self.state[12]
        _modify_gem5(
            self.tlb_root,
            "size\ =\ Param\.Int\(\d+,\ \"TLB\ size\"\)",
            "size = Param.Int(%d, \"TLB size\")" % (
                2 if tlb == 0 else \
                    round_power_of_two(tlb)
            )
        )

        # MSHR@D-Cache
        mshr = self.state[13]
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
            # cmd += "/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/scons "
            cmd += "scons "
            cmd += "build/RISCV/gem5.opt CCFLAGS_EXTRA=\"-I/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/include\" "
            cmd += "PYTHON_CONFIG=\"/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/python3-config\" "
            # cmd += "LDFLAGS_EXTRA=\"-L/research/dept8/gds/cbai/tools/protobuf-3.6.1/build/lib -L/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/lib\" "
            cmd += "LINKFLAGS_EXTRA=\"-L/research/dept8/gds/cbai/tools/protobuf-3.6.1/build/lib -L/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/lib\" "
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
            cmd = "cd {} && scons build/RISCV/gem5.opt -j{} ".format(
                self.macros["gem5-research-root"],
                multiprocessing.cpu_count()
            )
            cmd += "{} && mv -f build/RISCV/gem5.opt build/RISCV/{};".format(
                cmd,
                self.macros["simulator"]
            )
        execute(cmd)

        # check for the compilation
        if not if_exist(os.path.join(
                self.macros["gem5-research-root"],
                "build", "RISCV", self.macros["simulator"]
            )
        ):
            error("The gem5 simulation is failed to generate. Please check your environment"
                " to make sure you can compile GEM5 successfully!"
            )

    def get_results(self, benchmark):
        instructions, cycles = 0, 0
        misc_stats = OrderedDict()
        with open(os.path.join(
                self.macros["gem5-research-root"],
                benchmark, "stats.txt"
            ), 'r'
        ) as f:
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

    def simulate_impl(self, bmark):
        bmark_root = os.path.join(
            self.macros["gem5-research-root"],
            bmark
        )
        if if_exist(bmark_root):
            remove(bmark_root)

        cmd = "cd {}; {} --outdir={} {} ".format(
            self.macros["gem5-research-root"],
            os.path.join(
                self.macros["gem5-research-root"],
                "build", "RISCV", self.macros["simulator"]
            ),
            bmark_root,
            os.path.join(
                self.macros["gem5-research-root"],
                "configs", "example", "se.py"
            )
        )
        cmd += "--cmd={} ".format(os.path.join(
                self.macros["gem5-benchmark-root"],
                bmark + ".riscv"
            )
        )
        cmd += "--num-cpus=1 "
        cmd += "--cpu-type=TimingSimpleCPU "
        cmd += "--caches "
        cmd += "--cacheline_size=64 "
        cmd += " --l1d_size={}kB ".format(
            (((self.state[10] * self.state[11]) << 6)) >> 10
        )
        cmd += "--l1i_size={}kB ".format(
            (((self.state[3] * 64) << 6)) >> 10
        )
        cmd += "--l1d_assoc={} ".format(
            self.state[11]
        )
        cmd += "--l1i_assoc={} ".format(
            self.state[3]
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
        cmd += "--l2-hwp-type=TaggedPrefetcher"
        # simulate
        execute(cmd, logger=self.logger)
        instructions, cycles, misc_stats = \
            self.get_results(bmark)
        return instructions, cycles, misc_stats

    def simulate(self):
        ipc = 0

        threads = []
        for bmark in self.benchmarks:
            thread = WorkerThread(
                func=self.simulate_impl,
                args=(bmark,)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        for thread in threads:
            instructions, cycles, misc_stats = thread.get_output()
            if instructions == 0 or cycles == 0:
                # an error occurs in the simulation
                self.simulation_is_failed = True
                return 0
            self.incr_stats(misc_stats)
            ipc += (instructions / cycles)
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

        if self.simulation_is_failed:
            return power, area

        pool = ThreadPool(len(self.benchmarks))
        for bmark in self.benchmarks:
            bmark_root = os.path.join(
                self.macros["gem5-research-root"],
                bmark
            )
            mcpat_xml = os.path.join(
                bmark_root, "{}-{}.xml".format("Rocket", self.idx)
            )
            mcpat_report = os.path.join(
                bmark_root, "{}-{}.rpt".format("Rocket", self.idx)
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
                            self.macros["towards-automated-risc-v-microarchitecture-design-with-reinforcement-learning-root"],
                            "tools",
                            "gem5-mcpat-parser.py"
                        ),
                        self.configs["configs"],
                        os.path.join(
                            bmark_root, "config.json"
                        ),
                        os.path.join(
                            bmark_root, "stats.txt"
                        ),
                        os.path.join(
                            self.macros["towards-automated-risc-v-microarchitecture-design-with-reinforcement-learning-root"],
                            "tools",
                            "template",
                            "rocket.xml"
                        ),
                        ' '.join([str(s) for s in self.state]),
                        mcpat_xml,
                        os.path.join(
                            self.macros["towards-automated-risc-v-microarchitecture-design-with-reinforcement-learning-root"],
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
            bmark_root = os.path.join(
                self.macros["gem5-research-root"],
                bmark
            )
            mcpat_report = os.path.join(
                bmark_root, "{}-{}.rpt".format("Rocket", self.idx)
            )
            power += extract_power(mcpat_report)
            area += extract_area(mcpat_report)
        return power / len(self.benchmarks), \
            area / len(self.benchmarks)
