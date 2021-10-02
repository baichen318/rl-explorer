# Author: baichen318@gmail.com

import os
import sys
import re
import time
import multiprocessing
import numpy as np
from util import load_txt, execute, if_exist, write_txt, remove, mkdir, round_power_of_two
from vlsi.rocket.macros import MACROS

class VLSI(object):
    """ VLSI Flow """
    def __init__(self):
        super(VLSI, self).__init__()

    def steps(self):
        raise NotImplementedError()

    def run(self):
        for func in self.steps():
            func = getattr(self, func)
            func()


class BasicComponent(object):
    """ BasicComponent """
    def __init__(self, configs):
        super(BasicComponent, self).__init__()
        self.configs = configs
        self.btb = self.init_btb()
        self.icache = self.init_icache()
        self.mulDiv = self.init_mulDiv()
        self.dcache = self.init_dcache()

    def init_btb(self):
        return self.configs["basic-component"]["btb"]

    def init_icache(self):
        return self.configs["basic-component"]["icache"]

    def init_mulDiv(self):
        return self.configs["basic-component"]["mulDiv"]

    def init_dcache(self):
        return self.configs["basic-component"]["dcache"]


class PreSynthesizeSimulation(BasicComponent, VLSI):
    """ PreSynthesizeSimulation """
    # NOTICE: `counter` may be used in online settings
    counter = 0
    def __init__(self, configs, **kwargs):
        super(PreSynthesizeSimulation, self).__init__(configs)
        # a 10-dim vector: <torch.Tensor>
        self.rocket_configs = kwargs["rocket_configs"]
        self.soc_name = kwargs["soc_name"]
        self.core_name = kwargs["core_name"]
        self.batch = len(self.soc_name)
        # record every status of each design
        # -1: running, 0: success, -2: failure
        self.status = [-1 for i in range(self.batch)]

    def steps(self):
        return [
            "generate_design",
            "build_simv",
            "simulate"
        ]

    @staticmethod
    def set_tick(i, logger=None):
        # NOTICE: `idx` in *.yml refers to no. configs that begins to run
        PreSynthesizeSimulation.counter = i - 1
        if logger:
            logger.info("[INFO]: setting the idx: %d" % i)
        else:
            print("[INFO]: setting the idx: %d" % i)

    @staticmethod
    def tick():
        PreSynthesizeSimulation.counter += 1
        return PreSynthesizeSimulation.counter

    def __generate_mulDiv(self, idx):
        return """Some(MulDivParams(
          mulUnroll = %d,
          mulEarlyOut = %s,
          divEarlyOut = %s
        ))""" % (
            self.mulDiv[self.rocket_configs[idx][3]][0],
            "true" if self.mulDiv[self.rocket_configs[idx][3]][1] else "false",
            "true" if self.mulDiv[self.rocket_configs[idx][3]][2] else "false"
        )

    def __generate_fpu(self, idx):
        return "None" if self.rocket_configs[idx][2] == 0 else \
            "Some(FPUParams())"

    def __generate_btb(self, idx):
        return "None" if self.btb[self.rocket_configs[idx][0]][0] == 0 else \
        """Some(BTBParams(
          nEntries = %d,
          nRAS = %d,
          bhtParams = Some(BHTParams(nEntries=%d))
        )
      )""" % (
            self.btb[self.rocket_configs[idx][0]][1],
            self.btb[self.rocket_configs[idx][0]][0],
            self.btb[self.rocket_configs[idx][0]][2]
        )

    def __generate_dcache(self, idx):
        return """Some(DCacheParams(
          rowBits = site(SystemBusKey).beatBits,
          nSets = %d,
          nWays = %d,
          nTLBSets = 1,
          nTLBWays=%d,
          nMSHRs=%d,
          blockBytes = site(CacheBlockBytes)
        )
      )""" % (
                self.dcache[self.rocket_configs[idx][5]][0],
                self.dcache[self.rocket_configs[idx][5]][1],
                self.dcache[self.rocket_configs[idx][5]][2],
                self.dcache[self.rocket_configs[idx][5]][3]
            )

    def __generate_icache(self, idx):
        return """Some(ICacheParams(
          rowBits = site(SystemBusKey).beatBits,
          nSets = 64,
          nWays = %d,
          nTLBSets = 1,
          nTLBWays = %d,
          blockBytes = site(CacheBlockBytes)
        )
      )""" % (
                self.icache[self.rocket_configs[idx][1]][0],
                self.icache[self.rocket_configs[idx][1]][1],
            )

    def _generate_config_mixins(self):
        codes = []

        for idx in range(self.batch):
            codes.append('''
class %s(n: Int, overrideIdOffset: Option[Int] = None) extends Config((site, here, up) =>
{
  case RocketTilesKey => {
    val prev = up(RocketTilesKey, site)
    val idOffset = overrideIdOffset.getOrElse(prev.size)
    val rocket_core = RocketTileParams(
      core = RocketCoreParams(
        mulDiv = %s,
        fpu = %s,
        useVM = %s,
      ),
      btb = %s,
      dcache = %s,
      icache = %s)
    List.tabulate(n)(i => rocket_core.copy(hartId = i + idOffset)) ++ prev
  }
})
''' % (
        self.core_name[idx],
        self.__generate_mulDiv(idx),
        self.__generate_fpu(idx),
        "false" if self.rocket_configs[idx][4] == 0 else "true",
        self.__generate_btb(idx),
        self.__generate_dcache(idx),
        self.__generate_icache(idx)
    )
)

        return codes

    def _generate_rocket_configs(self):
        codes = []
        for idx in range(self.batch):
            codes.append('''
class %s extends Config(
  new freechips.rocketchip.subsystem.%s(1) ++
  new chipyard.config.AbstractConfig)
''' % (self.soc_name[idx], self.core_name[idx])
            )

        return codes

    def generate_design(self):
        self.configs["logger"].info("[INFO]: generate design...")
        codes = self._generate_config_mixins()
        with open(MACROS["config-mixins"], 'a') as f:
            f.writelines(codes)
        codes = self._generate_rocket_configs()
        with open(MACROS["rocket-configs"], 'a') as f:
            f.writelines(codes)

    def generate_scripts(self, idx):
        servers = [
            "hpc1", "hpc2", "hpc3", "hpc4", "hpc5",
            "hpc6", "hpc7", "hpc8", "hpc15", "hpc16"
        ]
        start = idx
        stride = self.batch // len(servers)
        remainder = self.batch % len(servers)

        for i in range(len(servers)):
            s = start
            e = start + stride - 1
            if e < s:
                continue
            else:
                execute(
                    "bash %s -s %d -e %d -x %s -f %s" % (
                        MACROS["generate-auto-vlsi-v2"],
                        s,
                        e,
                        MACROS["run-script"],
                        os.path.join(
                            MACROS["chipyard-vlsi-root"],
                            "rocket-compile-%s.sh" % servers[i]
                        )
                    )
                )
            start = e + 1
        if remainder > 0:
            # all in hpc16
            execute(
                "bash %s -s %d -e %d -x %s -f %s" % (
                    MACROS["generate-auto-vlsi-v2"],
                    start,
                    start + remainder - 1,
                    MACROS["run-script"],
                    os.path.join(
                        MACROS["chipyard-vlsi-root"],
                        "rocket-compile-hpc16.sh"
                    )
                )
            )

    def compile_and_simulate(self):
        # generate auto-vlsi.sh
        self.configs["logger"].info("[INFO]: generate auto-vlsi script...")
        execute(
            "bash %s -s %d -e %d -x %s -f %s" % (
                MACROS["generate-auto-vlsi-v1"],
                PreSynthesizeSimulation.counter - self.batch + 1,
                PreSynthesizeSimulation.counter,
                MACROS["sim-script"],
                os.path.join(
                    MACROS["chipyard-sims-root"],
                    "rocket-auto-vlsi.sh"
                )
            ),
            logger=self.configs["logger"]
        )
        os.chdir(MACROS["chipyard-sims-root"])
        # compile and simulate
        execute("bash rocket-auto-vlsi.sh", logger=self.configs["logger"])
        os.chdir(MACROS["rl-explorer-root"])

    def query_status(self):
        # when `query_status` is executed, all compilations is done and simulation shall be running
        self.configs["logger"].info("[INFO]: query status...")
        def _validate(x):
            if x == 0 or x == -2:
                # if a status equals to -2, we do not collect its results
                return True
            elif x == -1:
                return False

        def _query_status(status):
            def unexpected_behavior(f):
                # Notice: handle these unexpected behavior to make auto-vlsi more robust
                # when one of these unexpected behavior occurs, we need to re-compile and simulate
                soc_name = os.path.basename(os.path.dirname(f))
                # case #1
                if os.path.exists(f) and \
                    (execute("test -s %s" % f) != 0 or \
                     execute("test -s %s" % f.strip(".out") + ".log") != 0) and \
                    execute("ps aux | grep cbai | grep simv-chipyard-%s | grep -v grep" % soc_name) \
                        != 0:
                    # this may occur when simv is successfully generated but run failed without
                    # generating any output
                    self.configs["logger"].info("[WARN]: empty simulation result.")
                    return True
                # case #2
                if not os.path.isdir(os.path.join(MACROS["chipyard-sims-output-root"], soc_name)):
                    self.configs["logger"].info("[WARN]: output directory is not created.")
                    return True
                # case #3
                if not os.path.isdir(os.path.join(MACROS["chipyard-sims-root"], soc_name)):
                    # this may occur when simv is not generated successfully
                    self.configs["logger"].info("[WARN]: simv is not generated.")
                    return True
                # case #4
                if os.path.exists(f) and execute("grep -rn \"Text file busy\" %s" % f) == 0:
                    # this case may be covered by case # 1
                    self.configs["logger"].info("[WARN]: Text file busy.")
                    return True
                return False
                
            for idx in range(self.batch):
                if status[idx] == 0 or status[idx] == -2:
                    continue
                root = os.path.join(
                    MACROS["chipyard-sims-output-root"],
                    self.soc_name[idx]
                )
                # `s` is leveraged to record status of each benchmark
                s = [-1 for i in range(len(self.configs["benchmarks"]))]
                for i in range(len(self.configs["benchmarks"])):
                    f = os.path.join(root, self.configs["benchmarks"][i] + ".out")
                    if os.path.exists(f) and execute("grep -rn \"PASSED\" %s" % f) == 0:
                        s[i] = 0
                    elif os.path.exists(f) and execute("grep -rn \"hung\" %s" % f) == 0:
                        s[i] = -2
                        execute("ps -ef | grep \"simv-chipyard-%s +permissive\" | grep -v grep | awk \'{print $4}\' | xargs kill -9" % self.soc_name[idx])
                        execute("ps -ef | grep \"vcs\" | grep \"simv-chipyard-%s\" | grep -v grep | awk \'{print $5}\' | xargs kill -9" % self.soc_name[idx])
                    elif unexpected_behavior(f):
                        # this is an occasional case!
                        os.chdir(MACROS["chipyard-sims-root"])
                        # kill all related jobs
                        execute("ps -ef | grep \"simv-chipyard-%s +permissive\" | grep -v grep | awk \'{print $4}\' | xargs kill -9" % self.soc_name[idx])
                        execute("ps -ef | grep \"vcs\" | grep \"simv-chipyard-%s\" | grep -v grep | awk \'{print $5}\' | xargs kill -9" % self.soc_name[idx])
                        # clean all residual files
                        execute("rm -rf simv-chipyard-%s* %s %s" % (
                                self.soc_name[idx],
                                self.soc_name[idx],
                                os.path.join(MACROS["chipyard-sims-output-root"], self.soc_name[idx])
                            )
                        )
                        execute("make MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' CONFIG=%s" %
                            self.soc_name[idx]
                        )
                        execute("mkdir -p %s; mkdir -p %s; chmod +x simv-chipyard-%s;" % (
                                self.soc_name[idx],
                                os.path.join(MACROS["chipyard-sims-output-root"], self.soc_name[idx]),
                                self.soc_name[idx]
                            )
                        )
                        execute("mv -f simv-chipyard-%s* %s;" % (self.soc_name[idx], self.soc_name[idx]))
                        execute("cp -f %s %s; sed -i \"s/PATTERN/%s/g\" %s/sim.sh" % (
                                MACROS["sim-script"],
                                self.soc_name[idx],
                                self.soc_name[idx],
                                self.soc_name[idx]
                            )
                        )
                        execute("cd %s; bash sim.sh; cd -" % self.soc_name[idx])
                        # sleep 45s
                        time.sleep(45)
                        os.chdir(MACROS["rl-explorer-root"])
                        s[i] = -1
                    else:
                        s[i] = -1
                if -2 in s:
                    # failed
                    status[idx] = -2
                elif -1 in s:
                    # still running
                    status[idx] = -1
                else:
                    assert s.count(0) == len(s), "[ERROR]: s: %s is unexplected." % s
                    status[idx] = 0
            return status

        self.status = [-1 for i in range(self.batch)]
        # TODO: should we set the maximum time period?
        while not all(list(map(_validate, _query_status(self.status)))):
            time.sleep(15)

    def get_results(self):
        ipc = [0 for i in range(self.batch)]
        for idx in range(self.batch):
            if self.status[idx] == -2:
                self.configs["logger"].info("[WARN]: %s fails to simulate." % self.soc_name[idx])
                continue
            root = os.path.join(
                MACROS["chipyard-sims-output-root"],
                self.soc_name[idx]
            )
            _ipc = 0
            for bmark in self.configs["benchmarks"]:
                f = os.path.join(root, bmark + ".log")
                with open(f, 'r') as f:
                    cnt = f.readlines()
                    cycles, instructions = 0, 0
                    for line in cnt:
                        if "[INFO]" in line and "cycles" in line and "instructions" in line:
                            try:
                                cycles = re.search(r'\d+\ cycles', line).group()
                                cycles = int(cycles.split("cycles")[0])
                                instructions = re.search(r'\d+\ instructions', line).group()
                                instructions = int(instructions.split("instructions")[0])
                            except AttributeError:
                                continue
                        if cycles != 0:
                            __ipc = instructions / cycles
                            msg = "[INFO]: Configs.: %s, Benchmark: %s, IPC: %.8f" % (
                                self.soc_name[idx], bmark, __ipc
                            )
                            self.configs["logger"].info(msg)
                            _ipc += __ipc
                            break
            ipc[idx] = _ipc / len(self.configs["benchmarks"])
            msg = "[INFO]: Configs.: %s, IPC: %.8f" % (self.soc_name[idx], ipc[idx])
            self.configs["logger"].info(msg)
        return ipc


class Gem5Wrapper(BasicComponent):
    """Gem5Wrapper"""
    def __init__(self, configs, state, idx):
        super(Gem5Wrapper, self).__init__(configs)
        self.state = state
        self.idx = idx
        # NOTICE: a fixed structure
        self.root = os.path.join(
            MACROS["gem5-root"],
            str(self.idx),
            "gem5-research"
        )
        self.construct_link()
        mkdir(MACROS["temp-root"])

    def construct_link(self):
        self.root_btb = os.path.join(
            self.root,
            "src",
            "cpu",
            "pred",
            "BranchPredictor.py"
        )
        self.root_tlb = os.path.join(
            self.root,
            "src",
            "arch",
            "riscv",
            "RiscvTLB.py"
        )
        self.root_cache = os.path.join(
            self.root,
            "configs",
            "common",
            "Caches.py"
        )
        self.root_m5out = os.path.join(
            self.root,
            "m5out"
        )


    def modify_gem5(self):
        def _modify_gem5(src, pattern, target, count=0):
            cnt = open(src, "r+").read()
            with open(src, 'w') as f:
                f.write(re.sub(r"%s" % pattern, target, cnt, count))

        # RAS@btb
        _modify_gem5(
            self.root_btb,
            "RASSize\ =\ Param\.Unsigned\(\d+,\ \"RAS\ size\"\)",
            "RASSize = Param.Unsigned(%d, \"RAS size\")" % (1 if self.btb[self.state[0]][0] == 0 else \
                round_power_of_two(self.btb[self.state[0]][0])
            )
        )
        # BTB@btb
        _modify_gem5(
            self.root_btb,
            "BTBEntries\ =\ Param\.Unsigned\(\d+,\ \"Number\ of\ BTB\ entries\"\)",
            "BTBEntries = Param.Unsigned(%d, \"Number of BTB entries\")" % (2 if self.btb[self.state[0]][1] == 0 \
                else round_power_of_two(self.btb[self.state[0]][0])
            )
        )
        # TLB@D-Cache
        _modify_gem5(
            self.root_tlb,
            "size\ =\ Param\.Int\(\d+,\ \"TLB\ size\"\)",
            "size = Param.Int(%d, \"TLB size\")" % (2 if self.dcache[self.state[5]][2] == 0 else \
                round_power_of_two(self.dcache[self.state[5]][2])
            )
        )
        # MSHR@D-Cache
        _modify_gem5(
            self.root_cache,
            "mshrs\ =\ \d+",
            "mshrs = %d" % (1 if self.dcache[self.state[5]][3] == 0 else \
                round_power_of_two(self.dcache[self.state[5]][3])
            ),
            count=1
        )

    def generate_gem5(self):
        # NOTICE: commands are manually designed
        if os.popen("hostname").readlines()[0].strip() == "cuhk":
            cmd = "cd %s; " % self.root
            cmd += "/home/baichen/cbai/tools/Python-3.9.7/build/bin/scons "
            cmd += "build/RISCV/gem5.opt PYTHON_CONFIG=\"/home/baichen/cbai/tools/Python-3.9.7/build/bin/python3-config\" "
            cmd += "-j%d; " % multiprocessing.cpu_count()
            cmd += "cd -; "
        else:
            cmd = "cd %s; " % self.root
            cmd += "/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/scons "
            cmd += "build/RISCV/gem5.opt CCFLAGS_EXTRA=\"-I/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/include\" "
            cmd += "PYTHON_CONFIG=\"/research/dept8/gds/cbai/tools/Python-3.9.7/build/bin/python3-config\" "
            cmd += "LDFLAGS_EXTRA=\"-L/research/dept8/gds/cbai/tools/protobuf-3.6.1/build/lib -L/research/dept8/gds/cbai/tools/hdf5-1.12.0/build/lib\" "
            cmd += "-j%d; " % multiprocessing.cpu_count()
            cmd += "cd -"
        execute(cmd)

    def get_results(self):
        instructions, cycles = 0, 0
        with open(os.path.join(self.root_m5out, "stats.txt"), 'r') as f:
            cnt = f.readlines()
        for line in cnt:
            if line.startswith("simInsts"):
                instructions = int(line.split()[1])
            if line.startswith("system.cpu.numCycles"):
                cycles = int(line.split()[1])
        return instructions, cycles

    def simulate(self):
        for bmark in self.configs["benchmarks"]:
            remove(os.path.join(MACROS["temp-root"], "m5out-%s" % bmark))
        ipc = 0
        for bmark in self.configs["benchmarks"]:
            cmd = "cd %s; build/RISCV/gem5.opt configs/example/se.py " % self.root
            cmd += "--cmd=%s " % os.path.join(
                MACROS["gem5-benchmark-root"],
                "riscv-tests",
                bmark + ".riscv"
            )
            cmd += "--num-cpus=1 "
            cmd += "--cpu-type=TimingSimpleCPU "
            cmd += "--caches --l1d_size=%dkB --l1i_size=%dkB " % (
                round(int(self.dcache[self.state[5]][0] * self.dcache[self.state[5]][1] * 64 / 1024)),
                round(int(self.dcache[self.state[1]][0] * 64 * 64 / 1024))
            )
            cmd += "--sys-clock=2000000000Hz "
            cmd += "--cpu-clock=2000000000Hz "
            cmd += "--sys-voltage=6.3V "
            cmd += "--l2cache "
            cmd += "--l2_size=64MB "
            cmd += "--l2_assoc=8 "
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
                    self.root_m5out,
                    os.path.join(MACROS["temp-root"], "m5out-%s" % bmark)
                )
            )
        ipc /= len(self.configs["benchmarks"])
        return ipc


    def evaluate_perf(self):
        self.modify_gem5()
        self.generate_gem5()
        ipc = self.simulate()
        return ipc

    def evaluate_power_and_area(self):
        def extract_power(mcpat_report):
            p_subthreshold = re.compile(r"Subthreshold\ Leakage\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            p_gate = re.compile(r"Gate\ Leakage\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            p_dynamic = re.compile(r"Runtime\ Dynamic\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
            subthreshold, gate, dynamic = 0, 0, 0
            with open(mcpat_report, 'r') as rpt:
                rpt = rpt.read()
                try:
                    subthreshold = float(p_subthreshold.findall(rpt)[1][0])
                    gate = float(p_gate.findall(rpt)[1][0])
                    dynamic = float(p_dynamic.findall(rpt)[1][0])
                except Exception as e:
                    print("[ERROR]:", e)
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
                    print("[ERROR]:", e)
                    exit(1)
            return area

        for bmark in self.configs["benchmarks"]:
            mcpat_xml = os.path.join(
                MACROS["temp-root"],
                "m5out-%s" % bmark,
                "%s-%s.xml" % ("Rocket", self.idx)
            )
            mcpat_report = os.path.join(
                MACROS["temp-root"],
                "m5out-%s" % bmark,
                "%s-%s.rpt" % ("Rocket", self.idx)
            )
            remove(mcpat_xml)
            remove(mcpat_report)
            execute(
                "python2 %s -d %s -c %s -s %s -t %s --state %s -o %s" % (
                    os.path.join(MACROS["tools-root"], "gem5-mcpat-parser.py"),
                    self.configs["design"],
                    os.path.join(MACROS["temp-root"], "m5out-%s" % bmark, "config.json"),
                    os.path.join(MACROS["temp-root"], "m5out-%s" % bmark, "stats.txt"),
                    os.path.join(MACROS["tools-root"], "template", "rocket.xml"),
                    ' '.join([str(s) for s in self.state]),
                    mcpat_xml
                ),
                logger=self.configs["logger"]
            )
            execute(
                "%s -infile %s -print_level 5 > %s" % (
                    os.path.join(MACROS["mcpat-root"], "mcpat"),
                    mcpat_xml,
                    mcpat_report
                )
            )
        return extract_power(mcpat_report), extract_area(mcpat_report)


def test_online_vlsi(configs, state):
    """
        configs: <dict>
    """
    # TODO: power & area
    execute(
        "mkdir -p test"
    )
    MACROS["config-mixins"] = os.path.join(
        "test",
        "Configs.scala"
    )
    MACROS["rocket-configs"] = os.path.join(
        "test",
        "RocketConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"

    idx = [PreSynthesizeSimulation.tick() for i in range(state.shape[0])]

    vlsi_manager = PreSynthesizeSimulation(
        configs,
        rocket_configs=state,
        soc_name=[
            "Rocket%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dCores" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: [
        "generate_design",
        # "compile_and_simulate",
        # "query_status"
    ]
    vlsi_manager.run()
    vlsi_manager.get_results = lambda x=None: [np.random.randn() \
        for i in range(vlsi_manager.batch)]
    return vlsi_manager.get_results()

def online_vlsi(configs, state):
    """
        configs: <dict>
        state: <numpy.ndarray>
    """
    # TODO: power & area
    idx = [PreSynthesizeSimulation.tick() for i in range(state.shape[0])]

    vlsi_manager = PreSynthesizeSimulation(
        configs,
        rocket_configs=state,
        soc_name=[
            "Rocket%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dCores" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: [
        "generate_design",
        "compile_and_simulate",
        "query_status"
    ]
    vlsi_manager.run()
    return vlsi_manager.get_results()

def test_offline_vlsi(configs):
    """
        configs: <dict>
    """
    execute(
        "mkdir -p test"
    )
    MACROS["config-mixins"] = os.path.join(
        "test",
        "Configs.scala"
    )
    MACROS["rocket-configs"] = os.path.join(
        "test",
        "RocketConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"
    MACROS["chipyard-vlsi-root"] = "test"

    design_set = load_txt(configs["design-output-path"])
    if len(design_set.shape) == 1:
        design_set = np.expand_dims(design_set, axis=0)
    idx = [PreSynthesizeSimulation.tick() for i in range(design_set.shape[0])]
    vlsi_manager = PreSynthesizeSimulation(
        configs,
        rocket_configs=design_set,
        soc_name=[
            "Rocket%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dCores" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: ["generate_design"]
    vlsi_manager.run()
    vlsi_manager.generate_scripts(idx[0])

def offline_vlsi(configs):
    """
        configs: <dict>
    """
    # affect Configs.scala, RocketConfigs.scala and compile.sh
    design_set = load_txt(configs["design-output-path"])
    if len(design_set.shape) == 1:
        design_set = np.expand_dims(design_set, axis=0)
    idx = [PreSynthesizeSimulation.tick() for i in range(design_set.shape[0])]
    vlsi_manager = PreSynthesizeSimulation(
        configs,
        rocket_configs=design_set,
        soc_name=[
            "Rocket%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dCores" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: ["generate_design"]
    vlsi_manager.run()
    vlsi_manager.generate_scripts(idx[0])


def generate_ipc(configs, root):
    ipc = 0
    count = 0
    for bmark in configs["benchmarks"]:
        f = os.path.join(
            root,
            bmark + ".riscv",
            bmark + ".log"
        )
        if if_exist(f):
            with open(f, 'r') as f:
                cnt = f.readlines()
            for line in cnt:
                if "[INFO]" in line and "cycles" in line and "instructions" in line:
                    try:
                        cycles = re.search(r'\d+\ cycles', line).group()
                        cycles = int(cycles.split("cycles")[0])
                        instructions = re.search(r'\d+\ instructions', line).group()
                        instructions = int(instructions.split("instructions")[0])
                        ipc += (instructions / cycles)
                        count += 1
                    except AttributeError:
                        continue
    # average on all successful benchmarks
    if count == 0:
        print("[WARN]: %s is failed in simulation!" % root)
        return 0
    else:
        return ipc / count


def generate_power(configs, root):
    power = 0
    cnt = 0
    for bmark in configs["benchmarks"]:
        f = os.path.join(
            root,
            bmark,
            "reports",
            "vcdplus.power.avg.max.report"
        )
        if if_exist(f):
            with open(f, 'r') as f:
                for line in f.readlines():
                    # NOTICE: extract power of RocketTile
                    if "tile (RocketTile)" in line:
                        power += float(line.split()[-2])
                        cnt += 1
    # average on all successful benchmarks
    if cnt == 0:
        print("[WARN]: %s is failed in power measurement!" % root)
        return 0
    else:
        return power / cnt


def generate_area(configs, root):
    area = 0
    f = os.path.join(
        root,
        "reports",
        "final_area.rpt"
    )
    if if_exist(f):
        with open(f, 'r') as f:
            for line in f.readlines():
                if "RocketTile" in line:
                    area = float(line.split()[-1])
    return area


def generate_dataset(configs):
    def write_metainfo(path, dataset):
        print("[INFO]: writing to %s" % path)
        with open(path, 'w') as f:
            for data in dataset:
                f.write(data[0] + '\t')
                f.write(str(data[1]) + '\t')
                for _data in data[2]:
                    f.write(_data[0] + '\t')
                    f.write(str(_data[1]) + '\t')
                    f.write(str(_data[2]) + '\t')
                    f.write(str(_data[3]) + '\t')
                f.write("avg" + '\t')
                f.write(str(data[3]) + '\n')

    dataset = []
    design_set = load_txt(configs["design-output-path"])
    for i in range(1, configs["batch"] + 1):
        _dataset = np.array([])
        # add arch. feature
        _dataset = np.concatenate((_dataset, design_set[i - 1]))
        soc_name = "Rocket%dConfig" % i
        project_name = "chipyard.TestHarness.Rocket%dConfig-ChipTop" % i
        vlsi_root = os.path.join(
            MACROS["chipyard-vlsi-root"],
            "build",
            project_name
        )
        power_root = os.path.join(
            MACROS["power-root"],
            soc_name + "-power"
        )
        vlsi_sim_root = os.path.join(
            vlsi_root,
            "sim-syn-rundir"
        )
        vlsi_syn_root = os.path.join(
            vlsi_root,
            "syn-rundir"
        )
        # generate ipc
        _dataset = np.concatenate((_dataset, [generate_ipc(configs, vlsi_sim_root)]))
        # generate power
        _dataset = np.concatenate((_dataset, [generate_power(configs, power_root)]))
        # generate area
        _dataset = np.concatenate((_dataset, [generate_area(configs, vlsi_syn_root)]))
        dataset.append(_dataset)
    dataset = np.array(dataset)
    write_txt(configs["dataset-output-path"], dataset, fmt="%f")
