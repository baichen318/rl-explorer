# Author: baichen318@gmail.com

from macros import MACROS

class VLSI(object):
    """ VLSI Flow """
    def __init__(self, configs):
        super(VLSI, self).__init__()
        self.configs = configs

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


class PreSynthesizeSimulation(VLSI, BasicComponent):
    """ PreSynthesizeSimulation """
    def __init__(self, configs, **kwargs):
        super(PreSynthesizeSimulation, self).__init__(configs)
        # a 19-dim vector: <torch.Tensor>
        self.boom_configs = kwargs["boom_configs"]
        self.soc_name = kwargs["soc_name"]
        self.core_name = kwargs["core_name"]

    def steps(self):
        return [
            "generate_design",
            "build_simv",
            "simulate"
        ]

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
        return """
              Seq(
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d),
                IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue , dispatchWidth=%d)
              )
        """ % (
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
        return """
              numIntPhysRegisters = %d,
              numFpPhysRegisters = %d
        """ % (
                self.registers[self.boom_configs[9]][0],
                self.registers[self.boom_configs[9]][1]
            )

    def __generate_mulDiv(self):
        choice = self.boom_configs[13]
        if choice == 0:
            return "0",
        elif choice == 1:
            return "8",
        else:
            assert choice == 3
            return "XLen"

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

        return """
              Some(
                DCacheParams(rowBits = site(SystemBusKey).beatBits, nSets=%d, nWays=%d, nTLBSets=%d, nTLBWays=%d, nMSHRs=%d, replacementPolicy=%s)
              )
        """ % (
                self.dcache[self.boom_configs[15]][0],
                self.dcache[self.boom_configs[15]][1],
                self.dcache[self.boom_configs[15]][2],
                self.dcache[self.boom_configs[15]][3],
                self.dcache[self.boom_configs[15]][4],
                __generate_replacement_policy()
            )

    def __generate_icache(self):
        return """
              Some(
                ICacheParams(rowBits = site(SystemBusKey).beatBits, nSets=%d, nWays=%d, nTLBSets=%d, nTLBWays=%d, fetchBytes=%d)
              )
        """ % (
                self.icache[self.boom_configs[0]][0],
                self.icache[self.boom_configs[0]][1],
                self.icache[self.boom_configs[0]][2],
                self.icache[self.boom_configs[0]][3],
                self.icache[self.boom_configs[0]][4],
            )

    def __generate_system_bus_key(self):
        choice = self.boom_configs[7]
        if choice <= 2:
            return 8
        else:
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

    def _generate_boom_configs(self):
        codes = []
        codes.append('''
class %s extends Config(
  new boom.common.%s(1) ++
  new chipyard.config.AbstractConfig)
        ''' % (self.soc_name, self.core_name)
        )

    def generate_design(self):
        codes = self._generate_config_mixins()
        with open(MACROS["config-mixins"], 'a') as f:
            f.writelines(codes)
        codes = self._generate_boom_configs()
        with open(MACROS["boom-configs"], 'a') as f:
            f.writelines(codes)

    def build_simv(self):
        os.chdir(MACROS["chipyard-sims-root"])
        # compile & build
        os.system(
            "make \
            MACROCOMPILER_MODE='-l vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \
            CONFIG=%s" % self.soc_name
        )
        # post-handling
        os.system(
            "mkdir -p %s" % self.soc_name
        )
        os.system(
            "mkdir -p output/%s" % self.soc_name
        )
        os.system(
            "mv %s %s" % ("simv-chipyard-%s*" % self.soc_name, self.soc_name)
        )
        os.system(
            "cp %s %s" % (, self.soc_name)
        )
        os.chdir('-')

    def simulate(self):
        os.chdir(MACROS["chipyard-sims-root"])
        # pre-handling
        os.system(
            "sed -i 's/PATTERN/%s/g' sim.sh" % self.soc_name
        )
        # simulate
        for bmark in MACROS["benchmarks"]:
            os.system(
                "bash sim.sh %s &" % bmark
            )

