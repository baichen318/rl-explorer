# Author: baichen318@gmail.com
# set -ex

start=${start:-"1"}
end=${end:-"10"}
model=${model:-""}
file=${file:-"compile-default.sh"}

function set_env() {
    function handler() {
        exit 1
    }

    trap 'handler' SIGINT
}

function generate_vcs() {
    script="arr=`seq ${start} ${end}`
        for idx in ${arr[@]}
        do
            echo running ${idx}
            /opt2/synopsys/vcs/bin/vcs -full64 -line -timescale=1ns/10ps \
                -CC -I/opt/synopsys/vcs/include -CC -I/research/dept8/gds/cbai/demo/chipyard/riscv-tools-install/include -CC -I/research/dept8/gds/cbai/research/chipyard/tools/DRAMSim2 \
                -CC -std=c++11 /research/dept8/gds/cbai/research/chipyard/tools/DRAMSim2/libdramsim.a \
                /research/dept8/gds/cbai/demo/chipyard/riscv-tools-install/lib/libfesvr.a +lint=all,noVCDE,noONGS,noUI -error=PCWM-L -error=noZMMCM -timescale=1ns/10ps -quiet -q +rad +v2k +vcs+lic+wait +vc+list \
                -f /research/dept8/gds/cbai/research/chipyard/vlsi/generated-src/chipyard.TestHarness.BOOM${model}Config${idx}Config/sim_files.common.f \
                -sverilog +systemverilogext+.sv+.svi+.svh+.svt -assert svaext +libext+.sv +v2k +verilog2001ext+.v95+.vt+.vp +libext+.v -debug_pp \
                -debug_access+all \
                +incdir+/research/dept8/gds/cbai/research/chipyard/vlsi/generated-src/chipyard.TestHarness.BOOM${model}Config${idx}Config \
                /research/dept8/gds/cbai/research/chipyard/vlsi/generated-src/chipyard.TestHarness.BOOM${model}Config${idx}Config/chipyard.TestHarness.BOOM${model}Config${idx}Config.harness.v \
                +libext+.v /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/syn-rundir/ChipTop.mapped.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SIMPLE_RVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SIMPLE_LVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SIMPLE_SLVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SIMPLE_SRAM_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_AO_RVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_AO_LVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_AO_SLVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_AO_SRAM_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_OA_RVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_OA_LVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_OA_SLVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_OA_SRAM_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SEQ_RVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SEQ_LVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SEQ_SLVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_SEQ_SRAM_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_INVBUF_RVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_INVBUF_LVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_INVBUF_SLVT_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/tech-asap7-cache/extracted/ASAP7_PDKandLIB.tar/ASAP7_PDKandLIB_v1p5/asap7libs_24.tar.bz2/asap7libs_24/verilog/asap7sc7p5t_24_INVBUF_SRAM_TT.v \
                /research/dept8/gds/cbai/research/chipyard/vlsi/generated-src/chipyard.TestHarness.BOOM${model}Config${idx}Config/sram_behav_models.v \
                +define+VCS +define+CLOCK_PERIOD=0.5 +define+RESET_DELAY=777.7 +define+PRINTF_COND=TestDriver.printf_cond +define+STOP_COND=!TestDriver.reset \
                +define+RANDOMIZE_MEM_INIT +define+RANDOMIZE_REG_INIT +define+RANDOMIZE_GARBAGE_ASSIGN \
                +define+RANDOMIZE_INVALID_ASSIGN +define+DEBUG \
                -debug +notimingcheck +delay_mode_zero -top TestDriver \
                -P /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/sim-syn-rundir/access.tab \
                -o /research/dept8/gds/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.BOOM${model}Config${idx}Config-ChipTop/sim-syn-rundir/simv
        done
    "
    echo "generating vcs script: " ${file}
    echo ${script} > ${file}
}

while getopts "s:e:m:f:" arg
do
    case $arg in
        s)
            start=${OPTARG}
            ;;
        e)
            end=${OPTARG}
            ;;
        m)
            model=${OPTARG}
            ;;
        f)
            file=${OPTARG}
            ;;
        *)
            echo "Undefined parameters"
            exit
            ;;
    esac
done

set_env
generate_vcs
