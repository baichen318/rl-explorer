# Author: baichen318@gmail.com

import os
import sys
import re
import time
import numpy as np
from util import load_txt, execute, if_exist, write_txt
from vlsi.boom.macros import MACROS

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
        self.registers = self.init_registers()
        self.issue_unit = self.init_issue_unit()
        self.dcache = self.init_dcache()
        self.lsu = self.init_lsu()
        self.ifu_buffers = self.init_ifu_buffers()

    def init_registers(self):
        return self.configs["basic-component"]["registers"]

    def init_issue_unit(self):
        return self.configs["basic-component"]["issue-unit"]

    def init_dcache(self):
        return self.configs["basic-component"]["dcache"]

    def init_lsu(self):
        return self.configs["basic-component"]["lsu"]

    def init_ifu_buffers(self):
        return self.configs["basic-component"]["ifu-buffers"]


class PreSynthesizeSimulation(BasicComponent, VLSI):
    """ PreSynthesizeSimulation """
    # NOTICE: `counter` may be used in online settings
    counter = 0
    def __init__(self, configs, **kwargs):
        super(PreSynthesizeSimulation, self).__init__(configs)
        # a 10-dim vector: <torch.Tensor>
        self.boom_configs = kwargs["boom_configs"]
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

    def __generate_bpd(self, idx):
        choice = self.boom_configs[idx][0]
        if choice == 0:
            return "new WithTAGELBPD ++"
        elif choice == 1:
            return "new WithBoom2BPD ++"
        elif choice == 2:
            return "new WithAlpha21264BPD ++"
        else:
            assert choice == 3
            return "new WithSWBPD ++"

    def __generate_issue_unit(self, idx):
        return """Seq(
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue, dispatchWidth=%d)
              )""" % (
                # IQT_MEM
                1 if self.boom_configs[idx][4] <= 3 else 2,
                self.issue_unit[self.boom_configs[idx][7]][0],
                self.boom_configs[idx][4],
                # IQT_INT
                self.boom_configs[idx][4],
                self.issue_unit[self.boom_configs[idx][7]][1],
                self.boom_configs[idx][4],
                # IQT_FP
                1 if self.boom_configs[idx][4] <= 3 else 2,
                self.issue_unit[self.boom_configs[idx][7]][2],
                self.boom_configs[idx][4]
            )

    def __generate_registers(self, idx):
        return """numIntPhysRegisters = %d,
              numFpPhysRegisters = %d""" % (
                self.registers[self.boom_configs[idx][6]][0],
                self.registers[self.boom_configs[idx][6]][1]
            )

    def __generate_lsu(self, idx):
        return """numLdqEntries = %d,
              numStqEntries = %d""" % (
                self.lsu[self.boom_configs[idx][8]][0],
                self.lsu[self.boom_configs[idx][8]][1]
            )

    def __generate_ifu_buffers(self, idx):
        return """numFetchBufferEntries = %d,
              ftq = FtqParameters(nEntries=%d)""" % (
                self.ifu_buffers[self.boom_configs[idx][2]][0],
                self.ifu_buffers[self.boom_configs[idx][2]][1]
            )

    def __generate_mulDiv(self, idx):
        """ deprecated """
        choice = self.boom_configs[idx][13]
        if choice == 0:
            return "1"
        else:
            assert choice == 1
            return "8"

    def __generate_dcache(self, idx):
        def __generate_replacement_policy():
            """ deprecated """
            choice = self.boom_configs[idx][16]
            if choice == 0:
                return "random"
            elif choice == 1:
                return "lru"
            else:
                assert choice == 2, "[ERROR]: choice = %d, should be 2." % choice
                return "plru"

        return """Some(
                DCacheParams(
                  rowBits=site(SystemBusKey).beatBits,
                  nSets=64,
                  nWays=%d,
                  nTLBWays=%d,
                  nMSHRs=%d
                )
            )""" % (
                self.boom_configs[idx][1],
                self.dcache[self.boom_configs[idx][9]][1],
                self.dcache[self.boom_configs[idx][9]][0],
            )

    def __generate_icache(self, idx):
        """ deprecated """
        return """Some(
                ICacheParams(rowBits = site(SystemBusKey).beatBits, nSets=%d, nWays=%d, nTLBSets=%d, nTLBWays=%d, fetchBytes=%d)
            )""" % (
                self.icache[self.boom_configs[idx][0]][0],
                self.icache[self.boom_configs[idx][0]][1],
                self.icache[self.boom_configs[idx][0]][2],
                self.icache[self.boom_configs[idx][0]][3],
                self.icache[self.boom_configs[idx][0]][4],
            )

    def __generate_system_bus_key(self, idx):
        # fetchBytes
        choice = self.boom_configs[idx][1]
        if choice == 4:
            return 8
        else:
            assert choice == 8, "[ERROR] choice is %d" % choice
            return 16

    def _generate_config_mixins(self):
        codes = []

        for idx in range(self.batch):
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
              %s,
              maxBrCount = %d,
              %s,
              fpu = Some(
                freechips.rocketchip.tile.FPUParams(
                  sfmaLatency=4, dfmaLatency=4, divSqrt=true
                )
              )
            ),
            dcache = %s,
            icache = Some(
              ICacheParams(rowBits = site(SystemBusKey).beatBits, nSets=64, nWays=%d, fetchBytes=%d*4)
            ),
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
        self.core_name[idx],
        self.__generate_bpd(idx),
        self.boom_configs[idx][1],
        self.boom_configs[idx][4],
        self.boom_configs[idx][5],
        self.__generate_issue_unit(idx),
        self.__generate_registers(idx),
        self.__generate_lsu(idx),
        self.boom_configs[idx][3],
        self.__generate_ifu_buffers(idx),
        self.__generate_dcache(idx),
        self.boom_configs[idx][1],
        self.boom_configs[idx][1] // 2,
        self.__generate_system_bus_key(idx)
    )
)

        return codes

    def _generate_boom_configs(self):
        codes = []
        for idx in range(self.batch):
            codes.append('''
class %s extends Config(
  new boom.common.%s(1) ++
  new chipyard.config.AbstractConfig)
''' % (self.soc_name[idx], self.core_name[idx])
            )

        return codes

    def generate_design(self):
        self.configs["logger"].info("[INFO]: generate design...")
        codes = self._generate_config_mixins()
        with open(MACROS["config-mixins"], 'a') as f:
            f.writelines(codes)
        codes = self._generate_boom_configs()
        with open(MACROS["boom-configs"], 'a') as f:
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
                        MACROS["sim-script"],
                        os.path.join(
                            MACROS["chipyard-vlsi-root"],
                            "boom-auto-vlsi-%s.sh" % servers[i]
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
                    MACROS["sim-script"],
                    os.path.join(
                        MACROS["chipyard-vlsi-root"],
                        "boom-auto-vlsi-hpc16.sh"
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
                    "boom-auto-vlsi.sh"
                )
            ),
            logger=self.configs["logger"]
        )
        os.chdir(MACROS["chipyard-sims-root"])
        # compile and simulate
        execute("bash boom-auto-vlsi.sh", logger=self.configs["logger"])
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
        "config-mixins.scala"
    )
    MACROS["boom-configs"] = os.path.join(
        "test",
        "BoomConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"

    idx = [PreSynthesizeSimulation.tick() for i in range(state.shape[0])]

    vlsi_manager = PreSynthesizeSimulation(
        configs,
        boom_configs=state,
        soc_name=[
            "Boom%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dBooms" % i for i in idx
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
        boom_configs=state,
        soc_name=[
            "Boom%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dBooms" % i for i in idx
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
        "config-mixins.scala"
    )
    MACROS["boom-configs"] = os.path.join(
        "test",
        "BoomConfigs.scala"
    )
    MACROS["chipyard-sims-root"] = "test"
    MACROS["chipyard-vlsi-root"] = "test"

    design_set = load_txt(configs["design-output-path"])
    if len(design_set.shape) == 1:
        design_set = np.expand_dims(design_set, axis=0)
    idx = [PreSynthesizeSimulation.tick() for i in range(design_set.shape[0])]

    vlsi_manager = PreSynthesizeSimulation(
        configs,
        boom_configs=design_set,
        soc_name=[
            "Boom%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dBooms" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: ["generate_design"]
    vlsi_manager.run()
    vlsi_manager.generate_scripts(idx[0])

def offline_vlsi(configs):
    """
        configs: <dict>
    """
    # affect config-mixins.scala, BoomConfigs.scala and compile.sh
    design_set = load_txt(configs["design-output-path"])
    if len(design_set.shape) == 1:
        design_set = np.expand_dims(design_set, axis=0)
    idx = [PreSynthesizeSimulation.tick() for i in range(design_set.shape[0])]

    vlsi_manager = PreSynthesizeSimulation(
        configs,
        boom_configs=design_set,
        soc_name=[
            "Boom%dConfig" % i for i in idx
        ],
        core_name=[
            "WithN%dBooms" % i for i in idx
        ]
    )
    vlsi_manager.steps = lambda x=None: ["generate_design"]
    vlsi_manager.run()
    vlsi_manager.generate_scripts(idx[0])

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

