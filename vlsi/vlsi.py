# Author: baichen318@gmail.com

import os
from glob import glob
import numpy as np
from util import if_exist, execute, create_logger, mkdir, dump_yaml, read_csv, load_txt
from .macros import MACROS, modify_macros

class VLSI(object):
    def __init__(self, kwargs):
        self.configs = kwargs["configs"]
        self.idx = str(kwargs['idx'])
        self.core_name = "Config" + self.idx
        self.soc_name = "BOOM" + self.core_name + "Config"
        self.logger = kwargs['logger']
        if_exist(MACROS['config-mixins'], strict=True)
        if_exist(MACROS['boom-configs'], strict=True)
        modify_macros(self.core_name, self.soc_name)

        # variables used by `VLSI`
        self.latency = None
        self.power = None
        self.area = None

    def steps(self):
        return [
            'generate_design'
        ]

    def run(self):
        for func in self.steps():
            func = getattr(self, func)
            func()

    def generate_config_mixins(self):
        # fetchWidth decides Ftq_nEntries
        if self.configs[1] == 1:
            issueWidth = {
                "mem": 8,
                "int": 8,
                "fp": 8
            }
            Ftq_nEntries = 16
        elif self.configs[1] == 2 or self.configs[1] == 3:
            Ftq_nEntries = 32
            if self.configs[1] == 2:
                issueWidth = {
                    "mem": 12,
                    "int": 20,
                    "fp": 16
                }
            else:
                issueWidth = {
                    "mem": 16,
                    "int": 32,
                    "fp": 24
                }
        else:
            Ftq_nEntries = 40
            issueWidth = {
                "mem": 24,
                "int": 40,
                "fp": 32
            }
        codes = []

        codes.append('''
class %s extends Config((site, here, up) => {''' % self.core_name
        )

        # backbone
        codes.append('''
  case BoomTilesKey => up(BoomTilesKey, site) map { b => b.copy(
    core = b.core.copy(
      fetchWidth = %d, // fetchWidth
      decodeWidth = %d, // decodeWidth
      numRobEntries = %d, // numRobEntries
      issueParams = Seq(
        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_MEM.litValue, dispatchWidth=%d), // mem_issueWidth
        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_INT.litValue, dispatchWidth=%d), // int_issueWidth
        IssueParams(issueWidth=%d, numEntries=%d, iqType=IQT_FP.litValue , dispatchWidth=%d)), // fp_issueWidth
      numIntPhysRegisters = %d, // numIntPhysRegisters
      numFpPhysRegisters = %d, // numFpPhysRegisters
      numLdqEntries = %d, // numLdqEntries
      numStqEntries = %d, // numStqEntries
      maxBrCount = %d, // maxBrCount
      numFetchBufferEntries = %d, // numFetchBufferEntries
      numRasEntries = %d, // numRasEntries
      fpu = Some(freechips.rocketchip.tile.FPUParams(sfmaLatency=4, dfmaLatency=4, divSqrt=true)),
      ftq = FtqParameters(nEntries=%d),''' % (self.configs[0], self.configs[1],
            self.configs[3], self.configs[10], issueWidth["mem"], self.configs[1],
            self.configs[11], issueWidth["int"], self.configs[1], self.configs[12],
            issueWidth["fp"], self.configs[1], self.configs[5], self.configs[6],
            self.configs[7], self.configs[8], self.configs[9], self.configs[2],
            self.configs[4], Ftq_nEntries)
        )
        if self.configs[1] == 1:
            codes.append('''
      nPerfCounters = 2'''
            )
        elif self.configs[1] == 2:
            codes.append('''
      nPerfCounters = 6'''
            )
        elif self.configs[1] == 4:
            codes.append('''
      numDCacheBanks = 2,
      enablePrefetching = true'''
            )
        elif self.configs[1] == 5:
            codes.append('''
      numDCacheBanks = 1,
      enablePrefetching = true'''
            )
        codes.append('''
    ),'''
        )

        if self.configs[1] == 1 or self.configs[1] == 2:
            codes.append('''
    dcache = Some(
      DCacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets=64,
        nWays=%d, // DCacheParams_nWays
        nMSHRs=%d, // DCacheParams_nMSHRs
        nTLBEntries=%d // DCacheParams_nTLBEntries
      )
    ),
    icache = Some(
      ICacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets=64,
        nWays=%d, // ICacheParams_ nWays
        nTLBEntries=%d, // ICacheParams_nTLBEntries
        fetchBytes=%d*4 // ICacheParams_fetchBytes
      )
    )
  )}''' % (self.configs[13], self.configs[14], self.configs[15],
             self.configs[16], self.configs[17], self.configs[18])
            )
        elif self.configs[2] == 3:
            codes.append('''
    dcache = Some(
      DCacheParams(
        rowBits = site(SystemBusKey).beatBytes*8,
        nSets=64,
        nWays=%d, // DCacheParams_ nWays
        nMSHRs=%d, // DCacheParams_ nMSHRs
        nTLBEntries=%d // DCacheParams_ nTLBEntries
      )
    ),
    icache = Some(
      ICacheParams(
        rowBits = site(SystemBusKey).beatBytes*8,
        nSets=64,
        nWays=%d, // ICacheParams_ nWays
        nTLBEntries=%d, // ICacheParams_nTLBEntries
        fetchBytes=%d*4 // ICacheParams_ fetchBytes
      )
    )
  )}''' % (self.configs[13], self.configs[14], self.configs[15],
             self.configs[16], self.configs[17], self.configs[18])
            )
        else:
            codes.append('''
    dcache = Some(
      DCacheParams(
        rowBits = site(SystemBusKey).beatBytes*8,
        nSets=64,
        nWays=%d, // DCacheParams_ nWays
        nMSHRs=%d, // DCacheParams_ nMSHRs
        nTLBEntries=%d // DCacheParams_ nTLBEntries
      )
    ),
    icache = Some(
      ICacheParams(
        rowBits = site(SystemBusKey).beatBytes*8,
        nSets=64,
        nWays=%d, // ICacheParams_ nWays
        nTLBEntries=%d, // ICacheParams_nTLBEntries
        fetchBytes=%d*4, // ICacheParams_ fetchBytes
        prefetch=true
      )
    )
  )}''' % (self.configs[13], self.configs[14], self.configs[15],
             self.configs[16], self.configs[17], self.configs[18])
            )

        codes.append('''
  case SystemBusKey => up(SystemBusKey, site).copy(beatBytes = 8)
  case XLen => 64
  case MaxHartIdBits => log2Up(site(BoomTilesKey).size)'''
        )
        codes.append('''
})

'''
        )

        return codes

    def generate_boom_configs(self):
        codes = []
        codes.append('''
class %s extends Config(
  new chipyard.iobinders.WithUARTAdapter ++
  new chipyard.iobinders.WithTieOffInterrupts ++
  new chipyard.iobinders.WithBlackBoxSimMem ++
  new chipyard.iobinders.WithTiedOffDebug ++
  new chipyard.iobinders.WithSimSerial ++
  new testchipip.WithTSI ++
  new chipyard.config.WithBootROM ++
  new chipyard.config.WithUART ++
  new chipyard.config.WithL2TLBs(1024) ++
  new freechips.rocketchip.subsystem.WithNoMMIOPort ++
  new freechips.rocketchip.subsystem.WithNoSlavePort ++
  new freechips.rocketchip.subsystem.WithInclusiveCache ++
  new freechips.rocketchip.subsystem.WithNExtTopInterrupts(0) ++
  new boom.common.%s ++
  new boom.common.WithNBoomCores(1) ++
  new freechips.rocketchip.subsystem.WithCoherentBusTopology ++
  new freechips.rocketchip.system.BaseConfig)
        ''' % (self.soc_name, self.core_name))

        return codes

    def generate_design(self):
        codes = self.generate_config_mixins()
        with open(MACROS['config-mixins'], 'a') as f:
            f.writelines(codes)

        codes = self.generate_boom_configs()
        with open(MACROS['boom-configs'], 'a') as f:
            f.writelines(codes)

        self.logger.info("generate design done.")

    def compilation(self):
        cmd = "cp -f %s %s" % (
            os.path.join(MACROS["scripts"], "compile.sh"),
            MACROS["compile-script"]
        )
        execute(cmd, self.logger)
        cmd = "sed -i 's/PATTERN/%s/g' %s" % (self.soc_name, MACROS["compile-script"])
        execute(cmd, self.logger)

        os.chdir(MACROS["chipyard-vlsi-root"])
        cmd = "bash %s" % MACROS["compile-script"]
        execute(cmd, self.logger)
        os.chdir(os.path.join(MACROS["vlsi-root"], os.path.pardir))

        self.logger.info("compilation done.")
        
    def synthesis(self):
        if_exist(
            os.path.join(
                MACROS["syn-rundir"],
                "ChipTop.mapped.v"
            ),
            strict=True
        )

        self.logger.info("synthesis done.")

    def generate_simv(self):

        def pre_misc_work():
            cmd = "mv -f %s %s/" % (MACROS["hir-file"], MACROS["generated-src"])
            execute(cmd, self.logger)
            cmd = "cp -f %s %s/" % (MACROS["sram-vhdl"], MACROS["generated-src"])
            execute(cmd, self.logger)

        def post_misc_work():
            cmd = "mv -f %s %s" % (
                os.path.join(
                    MACROS["sim-syn-rundir"],
                    "dhrystone.riscv",
                    "dramsim2_ini"
                ),
                MACROS["sim-syn-rundir"]
            )
            execute(cmd, self.logger)
            # DANGER!
            cmd = "rm -fr %s" % os.path.join(MACROS["sim-syn-rundir"], "dhrystone.riscv")
            execute(cmd, self.logger)
            cmd = "cp -f %s %s/" % (
                os.path.join(
                    MACROS["scripts"],
                    "run.tcl"
                ),
                MACROS["sim-syn-rundir"]
            )
            execute(cmd, self.logger)
            cmd = "sed -i 's/PATTERN/%s/g' %s" % (
                self.soc_name,
                os.path.join(MACROS["sim-syn-rundir"], "run.tcl")
            )
            execute(cmd, self.logger)
            cmd = "mv -f %s %s/" % (
                os.path.join(MACROS["chipyard-vlsi-root"], "csrc"),
                MACROS["sim-syn-rundir"]
            )
            execute(cmd, self.logger)
            cmd = "mv -f %s %s/" % (
                os.path.join(MACROS["chipyard-vlsi-root"], "vc_hdrs.h"),
                os.path.join(MACROS["sim-syn-rundir"])
            )
            execute(cmd, self.logger)

            # re-link the *.so
            so_file = glob(
                os.path.join(
                    MACROS["sim-syn-rundir"],
                    "csrc",
                    "*.so"
                )
            )[0]
            # DANGER!
            cmd = "rm -f %s" % so_file
            execute(cmd, self.logger)
            cmd = "ln -s %s %s" % (
                os.path.join(
                    MACROS["sim-syn-rundir"],
                    "simv.daidir",
                    os.path.basename(so_file)
                ),
                so_file
            )
            execute(cmd, self.logger)

        pre_misc_work()

        pattern = "chipyard.TestHarness.%s-ChipTop" % self.soc_name

        cmd = "cp -f %s %s" % (
            os.path.join(MACROS["scripts"], "vcs.sh"),
            MACROS["simv-script"]
        )
        execute(cmd, self.logger)
        cmd = "sed -i 's/PATTERN/%s/g' %s" % (self.soc_name, MACROS["simv-script"])
        execute(cmd, self.logger)
        os.chdir(MACROS["chipyard-vlsi-root"])
        cmd = "bash %s" % MACROS["simv-script"]
        execute(cmd, self.logger)
        os.chdir(os.path.join(MACROS["vlsi-root"], os.path.pardir))

        if_exist(
            os.path.join(
                MACROS["sim-syn-rundir"],
                "simv"
            ),
            strict=True
        )

        post_misc_work()

        self.logger.info("generate simv done.")

    def simulation(self):
        cmd =  "cp -f %s %s" % (
            os.path.join(MACROS["scripts"], "ptpx.sh"),
            MACROS["sim-syn-rundir"]
        )
        execute(cmd, self.logger)

        mkdir(MACROS["sim-path"])
        mkdir(MACROS["power-path"])

        os.chdir(MACROS["sim-syn-rundir"])
        cmd = "bash ptpx.sh -s %s -t %s -p %s -r" % (
            MACROS["sim-path"],
            MACROS["temp-sim-path"],
            MACROS["power-path"]
        )
        execute(cmd, self.logger)
        os.chdir(os.path.join(MACROS["vlsi-root"], os.path.pardir))

        self.logger.info("simulation done.")

    def record(self):
        def generate_latency_yml():
            latency_dict = {
                "data-path": os.path.join(MACROS["chipyard-vlsi-root"], "build"),
                "mode": "latency",
                "output-path": MACROS["temp-latency-csv"],
                "config-name": ["chipyard.TestHarness.%s-ChipTop" % self.soc_name]
            }
            dump_yaml(MACROS["temp-latency-yml"], latency_dict)

        def generate_power_yml():
            power_dict = {
                "data-path": os.path.join(MACROS["power-root"]),
                "mode": "power",
                "output-path": MACROS["temp-power-csv"],
                "config-name": ["%s-benchmarks" % self.core_name]
            }
            dump_yaml(MACROS["temp-power-yml"], power_dict)

        def generate_area_yml():
            area_dict = {
                "data-path": os.path.join(MACROS["chipyard-vlsi-root"], "build"),
                "mode": "area",
                "output-path": MACROS["temp-area-csv"],
                "config-name": ["chipyard.TestHarness.%s-ChipTop" % self.soc_name]
            }
            dump_yaml(MACROS["temp-area-yml"], area_dict)

        # Latency report
        generate_latency_yml()
        cmd = "python handle-data.py -c %s" % MACROS["temp-latency-yml"]
        execute(cmd, self.logger)
        # Power report
        generate_power_yml()
        cmd = "python handle-data.py -c %s" % MACROS["temp-power-yml"]
        execute(cmd, self.logger)
        # Area report
        generate_area_yml()
        cmd = "python handle-data.py -c %s" % MACROS["temp-area-yml"]
        execute(cmd, self.logger)

        latency = read_csv(MACROS["temp-latency-csv"])
        t = 0
        cnt = 0
        for v in latency:
            if not np.isnan(v[1]):
                t += v[1]
                cnt += 1
        t /= cnt
        self.latency = t

        power = read_csv(MACROS["temp-power-csv"])
        t = 0
        cnt = 0
        for v in power:
            if not np.isnan(v[-1]):
                t += v[-1]
                cnt += 1
        t /= cnt
        self.power = t

        self.area = read_csv(MACROS["temp-area-csv"])[0][1]

        # for debugging
        # self.latency = 0
        # self.power = 0
        # self.area = 0

    def clean(self):
        # DANGER!
        cmd = "rm -f %s" % MACROS["compile-script"]
        execute(cmd, self.logger)
        cmd = "rm -f %s" % MACROS["simv-script"]
        execute(cmd, self.logger)
        cmd = "rm -f %s" % MACROS["temp-latency-yml"]
        execute(cmd, self.logger)
        cmd = "rm -f %s" % MACROS["temp-power-yml"]
        execute(cmd, self.logger)
        cmd = "rm -f %s" % MACROS["temp-area-yml"]
        execute(cmd, self.logger)
        # cmd = "rm -f %s" % MACROS["temp-latency-csv"]
        # execute(cmd, self.logger)
        # cmd = "rm -f %s" % MACROS["temp-power-csv"]
        # execute(cmd, self.logger)
        # cmd = "rm -f %s" % MACROS["temp-area-csv"]
        # execute(cmd, self.logger)

def vlsi_flow(kwargs, queue=None):
    vlsi = VLSI(kwargs)
    vlsi.run()

    ret =  {
        "latency": vlsi.latency,
        "power": vlsi.power,
        "area:": vlsi.area
    }

    if queue:
        queue.put(ret)
    else:
        return ret

def offline_vlsi_flow():
    if_exist(
        configs["sample-output-path"],
        strict=True
    )
    dataset = load_txt(configs["sample-output-path"])

    for idx, data in enumerate(dataset):
        kwargs = {
            "configs": data,
            "idx": idx,
            "logger": create_logger("logs", "vlsi"),
        }
        vlsi = VLSI(kwargs)
        vlsi.run()

if __name__ == "__main__":
    argv = parse_args()
    configs = get_config(argv)
    offline_vlsi_flow()
