# Author: baichen318@gmail.com
import os
import sys
import re
import time
import numpy as np
from util import load_txt, execute, if_exist, write_txt
from macros import MACROS

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
        self.icache = self.init_icache()
        self.registers = self.init_registers()
        self.issue_unit = self.init_issue_unit()
        self.dcache = self.init_dcache()

    def init_icache(self):
        return self.configs["basic-component"]["icache"]

    def init_registers(self):
        return self.configs["basic-component"]["registers"]

    def init_issue_unit(self):
        return self.configs["basic-component"]["issue-unit"]

    def init_dcache(self):
        return self.configs["basic-component"]["dcache"]


class PreSynthesizeSimulation(BasicComponent, VLSI):
    """ PreSynthesizeSimulation """
    # NOTICE: `counter` may be used in online settings
    counter = 1
    def __init__(self, configs, **kwargs):
        super(PreSynthesizeSimulation, self).__init__(configs)
        # a 19-dim vector: <torch.Tensor>
        self.boom_configs = kwargs["boom_configs"]
        self.soc_name = kwargs["soc_name"]
        self.core_name = kwargs["core_name"]
        self.tick()

    def steps(self):
        return [
            "generate_design",
            "build_simv",
            "simulate"
        ]

    def tick(self):
        PreSynthesizeSimulation.counter += 1

    def __generate_bpd(self):
        choice = self.boom_configs[4]
        if choice == 0:
            return "new WithTAGELBPD ++"
        elif choice == 1:
            return "new WithBoom2BPD ++"
        elif choice == 2:
            return "new WithAlpha21264BPD ++"
        else:
            assert choice == 3
            return "new WithSWBPD ++"

    def __generate_issue_unit(self):
        return """Seq(
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue , dispatchWidth=%d)
              )""" % (
                self.boom_configs[10],
                self.issue_unit[self.boom_configs[12]][1],
                self.boom_configs[7],
                self.boom_configs[7],
                self.issue_unit[self.boom_configs[12]][0],
                self.boom_configs[7],
                self.boom_configs[11],
                self.issue_unit[self.boom_configs[12]][2],
                self.boom_configs[7]
            )

    def __generate_registers(self):
        return """numIntPhysRegisters = %d,
              numFpPhysRegisters = %d""" % (
                self.registers[self.boom_configs[9]][0],
                self.registers[self.boom_configs[9]][1]
            )

    def __generate_mulDiv(self):
        choice = self.boom_configs[13]
        if choice == 0:
            return "1"
        else:
            assert choice == 1
            return "8"

    def __generate_dcache(self):
        def __generate_replacement_policy():
            choice = self.boom_configs[16]
            if choice == 0:
                return "random"
            elif choice == 1:
                return "lru"
            else:
                assert choice == 2
                return "plru"

        return """Some(
                DCacheParams(
                  rowBits = site(SystemBusKey).beatBits,
                  nSets=%d,
                  nWays=%d,
                  nTLBSets=%d,
                  nTLBWays=%d,
                  nMSHRs=%d,
                  replacementPolicy="%s"
                )
            )""" % (
                self.dcache[self.boom_configs[15]][0],
                self.dcache[self.boom_configs[15]][1],
                self.dcache[self.boom_configs[15]][2],
                self.dcache[self.boom_configs[15]][3],
                self.dcache[self.boom_configs[15]][4],
                __generate_replacement_policy()
            )

    def __generate_icache(self):
        return """Some(
                ICacheParams(rowBits = site(SystemBusKey).beatBits, nSets=%d, nWays=%d, nTLBSets=%d, nTLBWays=%d, fetchBytes=%d)
            )""" % (
                self.icache[self.boom_configs[0]][0],
                self.icache[self.boom_configs[0]][1],
                self.icache[self.boom_configs[0]][2],
                self.icache[self.boom_configs[0]][3],
                self.icache[self.boom_configs[0]][4],
            )

    def __generate_system_bus_key(self):
        # fetchBytes
        choice = self.icache[self.boom_configs[0]][4]
        if choice == 8:
            return 8
        else:
            assert choice == 16
            return 16

    def _generate_config_mixins(self):
        codes = []

        codes.append('''
class %s(n: Int = 1, overrideIdOffset: Option[Int] = None) extends Config(
  %s
  new Config((site, here, up) => {
    case TilesLocated(InSubsystem) => {
      val prev = up(TilesLocated(InSubsystem), site)
      val idOffset = overrideIdOffset.getOrElse(prev.size)
      (0 until n).map { i =>
        BoomTileAttachParams(
          tileParams = BoomTileParams(
            core = BoomCoreParams(
              fetchWidth = %d,
              decodeWidth = %d,
              numRobEntries = %d,
              issueParams = %s,
              %s,
              numLdqEntries = %d,
              numStqEntries = %d,
              maxBrCount = %d,
              numFetchBufferEntries = %d,
              enablePrefetching = true,
              numDCacheBanks = 1,
              ftq = FtqParameters(nEntries=%d),
              fpu = Some(
                freechips.rocketchip.tile.FPUParams(
                  sfmaLatency=4, dfmaLatency=4, divSqrt=true
                )
              ),
              mulDiv = Some(
                MulDivParams(
                  mulUnroll=%s,
                  mulEarlyOut=true,
                  divEarlyOut=true
                )
              )
            ),
            dcache = %s,
            icache = %s,
            hartId = i + idOffset
          ),
          crossingParams = RocketCrossingParams()
        )
      } ++ prev
    }
    case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = %d)
    case XLen => 64
  })
)
''' % (
        self.core_name,
        self.__generate_bpd(),
        self.boom_configs[1],
        self.boom_configs[7],
        self.boom_configs[8],
        self.__generate_issue_unit(),
        self.__generate_registers(),
        self.boom_configs[14],
        self.boom_configs[14],
        self.boom_configs[6],
        self.boom_configs[2],
        self.boom_configs[5],
        self.__generate_mulDiv(),
        self.__generate_dcache(),
        self.__generate_icache(),
        self.__generate_system_bus_key()
    )
)

        return codes

    def _generate_boom_configs(self):
        codes = []
        codes.append('''
class %s extends Config(
  new boom.common.%s(1) ++
  new chipyard.config.AbstractConfig)
''' % (self.soc_name, self.core_name)
        )

        return codes

    def generate_design(self):
        codes = self._generate_config_mixins()
        with open(MACROS["config-mixins"], 'a') as f:
            f.writelines(codes)
        codes = self._generate_boom_configs()
        with open(MACROS["boom-configs"], 'a') as f:
            f.writelines(codes)

    def generate_scripts(self, batch, start):
        servers = [
            "hpc1", "hpc2", "hpc3", "hpc4", "hpc5",
            "hpc6", "hpc7", "hpc8", "hpc15", # "hpc16"
        ]
        stride = batch // len(servers)
        remainder = batch % len(servers)

        for i in range(len(servers)):
            s = start
            e = start + stride - 1
            if e < s:
                continue
            else:
                execute(
                    "bash vlsi/scripts/compile.sh -s %d -e %d -x %s -f %s" % (
                        s,
                        e,
                        MACROS["sim-script"],
                        os.path.join(
                            MACROS["chipyard-sims-root"],
                            "compile-%s.sh" % servers[i]
                        )
                    )
                )
            start = e + 1
        if remainder > 0:
            # all in hpc16
            execute(
                "bash vlsi/scripts/compile.sh -s %d -e %d -x %s -f %s" % (
                    start,
                    start + remainder - 1,
                    MACROS["sim-script"],
                    os.path.join(
                        MACROS["chipyard-sims-root"],
                        "compile-hpc16.sh"
                    )
                )
            )


    def build_simv(self):
        os.chdir(MACROS["chipyard-sims-root"])
        # compile & build
        execute(
            "make \
            MACROCOMPILER_MODE='-l vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \
            CONFIG=%s" % self.soc_name
        )
        # post-handling
        execute(
            "mkdir -p %s" % self.soc_name
        )
        execute(
            "mkdir -p output/%s" % self.soc_name
        )
        execute(
            "mv %s %s" % ("simv-chipyard-%s*" % self.soc_name, self.soc_name)
        )
        execute(
            "cp %s %s" % (
                MACROS["sim-script"],
                os.path.join(MACROS["chipyard-sims-root"], self.soc_name)
            )
        )
        os.chdir(MACROS["rl-explorer-root"])

    def simulate(self):
        os.chdir(
            os.path.join(
                MACROS["chipyard-sims-root"],
                self.soc_name
            )
        )
        # pre-handling
        execute(
            "sed -i 's/PATTERN/%s/g' sim.sh" % self.soc_name
        )
        # simulate
        execute(
            "bash sim.sh"
        )

    def query_status(self):
        def _query_status():
            root = os.path.join(
                MACROS["chipyard-sims-output-root"],
                self.soc_name
            )
            for bmark in self.configs["benchmarks"]:
                f = os.path.join(root, bmark + ".out")
                if os.path.exists(f) and execute("grep -rn \"PASSED\" %s" % f) == 0:
                    continue
                else:
                    return False
            return True

        # TODO: should we set the maximum time period?
        print("[INFO]:", "querying results...")
        while not _query_status():
            time.sleep(60)

    def get_results(self):
        root = os.path.join(
            MACROS["chipyard-sims-output-root"],
            self.soc_name
        )
        ipc = 0
        for bmark in self.configs["benchmarks"]:
            f = os.path.join(root, bmark + ".log")
            with open(f, 'r') as f:
                cnt = f.readlines()
                for line in cnt:
                    if "[INFO]" in line and "cycles" in line and "instructions" in line:
                        try:
                            cycles = re.search(r'\d+\ cycles', line).group()
                            cycles = int(cycles.split("cycles")[0])
                            instructions = re.search(r'\d+\ instructions', line).group()
                            instructions = int(instructions.split("instructions")[0])
                        except AttributeError:
                            continue
                    if "cycles" in locals() and "instructions" in locals():
                        ipc += instructions / cycles
                        del cycles
                        del instructions
        return ipc / len(self.configs["benchmarks"])


def test_offline_vlsi(configs):
    """
        configs: <dict>
    """
    design_set = load_txt(configs["design-output-path"])

    execute(
        "rm -fr test"
    )
    execute(
        "mkdir -p test"
    )
    MACROS["config-mixins"] = os.path.join(
        "test",
        "config-mixins.scala"
    )
    MACROS["boom-configs"] = os.path.join(
        "test",
        "BoomConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"

    idx = configs["idx"]
    for design in design_set:
        vlsi_manager = PreSynthesizeSimulation(
            configs,
            boom_configs=design,
            soc_name="Boom%dConfig" % idx,
            core_name="WithN%dBooms" % idx
        )
        vlsi_manager.steps = lambda x=None: ["generate_design"]
        vlsi_manager.run()
        idx = idx + 1

    vlsi_manager.generate_scripts(len(design_set), configs["idx"])

def offline_vlsi(configs):
    """
        configs: <dict>
    """
    # affect config-mixins.scala, BoomConfigs.scala and compile.sh
    design_set = load_txt(configs["design-output-path"])

    idx = configs["idx"]
    for design in design_set:
        vlsi_manager = PreSynthesizeSimulation(
            configs,
            boom_configs=design,
            soc_name="Boom%dConfig" % idx,
            core_name="WithN%dBooms" % idx
        )
        vlsi_manager.steps = lambda x=None: ["generate_design"]
        vlsi_manager.run()
        idx = idx + 1

    vlsi_manager.generate_scripts(len(design_set), configs["idx"])

def test_online_vlsi(configs, state):
    """
        configs: <dict>
    """
    execute(
        "rm -fr test"
    )
    execute(
        "mkdir -p test"
    )
    MACROS["config-mixins"] = os.path.join(
        "test",
        "config-mixins.scala"
    )
    MACROS["boom-configs"] = os.path.join(
        "test",
        "BoomConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"

    for _state in state:
        vlsi_manager = PreSynthesizeSimulation(
            configs,
            boom_configs=_state,
            soc_name="Boom%dConfig" % PreSynthesizeSimulation.counter,
            core_name="WithN%dBooms" % PreSynthesizeSimulation.counter
        )
        vlsi_manager.steps = lambda x=None: [
            "generate_design",
            # "build_simv",
            # "simulate",
            # "query_status"
        ]
        vlsi_manager.run()
    vlsi_manager.get_results = lambda x=None: np.random.randn()
    return vlsi_manager.get_results()

def online_vlsi(configs, state):
    """
        configs: <dict>
        state: <numpy.ndarray>
    """
    for _state in state:
        vlsi_manager = PreSynthesizeSimulation(
            configs,
            boom_configs=_state,
            soc_name="Boom%dConfig" % PreSynthesizeSimulation.counter,
            core_name="WithN%dBooms" % PreSynthesizeSimulation.counter
        )
        vlsi_manager.steps = lambda x=None: [
            "generate_design",
            "build_simv",
            "simulate",
            "query_status"
        ]
    vlsi_manager.run()
    return vlsi_manager.get_results()

def _generate_dataset(configs, design_set, dataset, dir_n):
    # get feature vector `fv`
    idx = dir_n.strip("Boom").strip("Config")
    fv = list(design_set[int(idx) - 1])
    # get IPC
    _dataset = []
    ipc = 0
    for bmark in configs["benchmarks"]:
        f = os.path.join(
            MACROS["chipyard-sims-output-root"],
            dir_n,
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
                    except AttributeError:
                        continue
            if "cycles" in locals() and "instructions" in locals():
                ipc += instructions / cycles
                _dataset.append([bmark, instructions, cycles, ipc])
                del cycles, instructions
    if len(_dataset) > 0:
        dataset.append([idx, fv, _dataset, ipc / len(_dataset)])

def generate_dataset(configs):
    def _filter(dataset):
        _dataset = []
        for data in dataset:
            temp = []
            for i in data[1]:
                temp.append(i)
            temp.append(data[-1])
            _dataset.append(temp)

        return np.array(_dataset)

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

    design_set = load_txt(configs["design-output-path"])
    dataset = []
    for dir_n in os.listdir(MACROS["chipyard-sims-output-root"]):
        try:
            re.match(r'Boom\d+Config', dir_n).group()
            _generate_dataset(configs, design_set, dataset, dir_n)
        except AttributeError:
            continue
    write_txt(configs["data-output-path"], _filter(dataset), fmt="%f")
    # save more info.

    write_metainfo(
        os.path.join(
            os.path.dirname(configs["data-output-path"]),
            os.path.basename(configs["data-output-path"]).split(".")[0] + ".info"
        ),
        dataset
    )
