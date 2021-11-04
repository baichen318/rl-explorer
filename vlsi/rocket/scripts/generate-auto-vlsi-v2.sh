#!/bin/bash
# Author: baichen318@gmail.com

start=${start:-"1"}
end=${end:-"32"}
run_script=${run_script:-""}
file=${file:-""}

function set_env() {
    function handler() {
        exit 1
    }
    trap 'handler' SIGINT
}

function generate_auto_vlsi_v2() {
    echo "[INFO]: generating compilation script: " ${file}
cat > ${file} << EOF
#!/bin/bash
# Author: baichen318@gmail.com
# post-syn. auto-vlsi
# Auto-generated by ${BASH_SOURCE[0]}

power="/research/dept8/gds/cbai/research/synopsys-flow/build/pt-pwr"
benchmarks=(median towers vvadd multiply)

function compile() {
    soc_name=\${1}
    echo "[INFO] compile \${soc_name}"
    make build \\
        MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
        CONFIG=\${soc_name} \\
        BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
    # 300 sec. would be suitable
    sleep 300
}

function syn2sim() {
    soc_name=\${1}
    echo "[INFO] syn2sim \${soc_name}"
    make sim-syn \\
        MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
        CONFIG=\${soc_name} \\
        BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv
}

function sim() {
    soc_name=\${1}
    project_name=\${2}
    echo "[INFO] sim \${soc_name}"
    cp -f /research/dept8/gds/cbai/research/chipyard/vlsi/TestDriver/TestDriver.v generated-src/\${project_name}/
    rm -rf build/\${project_name}-ChipTop/sim-syn-rundir/simv*
    rm -rf build/\${project_name}-ChipTop/sim-syn-rundir/csrc build/\${project_name}-ChipTop/sim-syn-rundir/vc_hdrs.h
    make sim \\
        MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
        CONFIG=\${soc_name} \\
        BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
    # 180 sec. would be suitable
    sleep 180
}

function ptpx_impl() {
    soc_name=\${1}
    project_name=\${2}
    bmark=\${3}
    mv -f vcdplus.saif /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/
    power_name=\${power}/\${soc_name}-power
    mkdir -p \${power_name}
    cd \${power}
    make build_pt_dir=\${power_name}/"build-pt-"\${bmark} \\
        cur_build_pt_dir=\${power_name}/"current-pt-"\${bmark} \\
        vcs_dir=/research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv \\
        icc_dir=/research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/syn-rundir/
    mv -f \${power_name}/build-pt-\${bmark} \${power_name}/\${bmark}
    rm -rf \${power_name}/current-pt-\${bmark}
    rm -f /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/*.saif
    cd -
}

function sim2power() {
    soc_name=\${1}
    project_name=\${2}
    for bmark in \${benchmarks[@]}
    do
        echo benchmark: \${bmark}
        # create run.tcl
        cp -f ${run_script} /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/
        sed -i "s/PATTERN/\${soc_name}/g" /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/run.tcl
        mkdir -p /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv
        (set -o pipefail && /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/simv \\
            +permissive \\
            +dramsim \\
            +dramsim_ini_dir=/research/dept8/gds/cbai/research/chipyard/generators/testchipip/src/main/resources/dramsim2_ini \\
            +max-cycles=635000  \\
            -ucli -do /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/run.tcl \\
            +ntb_random_seed_automatic \\
            +verbose \\
            +saif=/research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}/vcdplus.saif \\
            +permissive-off \\
            /research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/\${bmark}.riscv </dev/null 2> \\
            >(spike-dasm > /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.out) | tee /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.log &)
        # 100: make sure that vcdplus.saif is generated sequentially!
        sleep 100
    done

    pass_bmark=()
    all_done=0
    while [[ \${all_done} == 0 ]]
    do
        for bmark in \${benchmarks[@]}
        do
            ret=\`grep "PASSED" /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.out\`
            if [[ ! -z \${ret} ]] && [[ ! \${pass_bmark[@]} =~ \${bmark} ]]
            then
                # waiting to generate saif
                while [[ -e vcdplus.saif ]]
                do
                    pass_bmark[\${#pass_bmark[*]}]=\${bmark}
                    ptpx_impl \${soc_name} \${project_name} \${bmark} &
                    break
                done
            fi
        done
        c=0
        for bmark in \${benchmarks[@]}
        do
            if [[ ! \${pass_bmark[@]} =~ \${bmark} ]]
            then
                all_done=0
            else
                c=\`expr \${c} + 1\`
            fi
            if [[ \${c} == \${#benchmarks[*]} ]]
            then
                all_done=1
            fi
        done
    done
}

function check_power_report() {
    # 1 means success, 0 means failure
    soc_name=\${1}-power
    success_bmark=()
    for bmark in \${benchmarks[@]}
    do
        path=\${power}/\${soc_name}
        if [[ -e \${path}/\${bmark}/reports/vcdplus.power.avg.max.report ]] && \\
            [[ ! \${success_bmark[@]} =~ \${bmark} ]]
        then
            success_bmark[\${#success_bmark[*]}]=\${bmark}
        fi
    done
    if [[ \${#success_bmark[*]} == \${#benchmarks[*]} ]]
    then
        return 1
    else
        return 0
    fi
}

# syn2sim
# verify all simv have been generated
count=\`expr ${end} - ${start} + 1\`
arr=\`seq ${start} ${end}\`
for idx in \${arr[@]}
do
    soc_name=Rocket\${idx}Config
    syn2sim \${soc_name}
    sleep 800
done

exit 0

# re-check compile
c=0
pass_config=()
while [[ \${c} -lt \${count} ]]
do
    for idx in \${arr[@]}
	do
        soc_name=Rocket\${idx}Config
        project_name=chipyard.TestHarness.\${soc_name}
        if [[ -e generated-src/\${project_name}/\${project_name}.top.v ]] && \\
            [[ ! \${pass_config[@]} =~ \${soc_name} ]]
        then
            c=\`expr \${c} + 1\`
            pass_config[\${#pass_config[*]}]=\${soc_name}
        else
            # check whether we still have this process
            ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
            ret=\$?
            if [[ \${ret} != 0 ]]
            then
                # the process is terminated
                compile \${soc_name}
            fi
        fi
    done
done

# syn.
for idx in \${arr[@]}
do
    soc_name=Rocket\${idx}Config
    syn2sim \${soc_name}
done


# re-check syn.
c=0
pass_config=()
while [[ \${c} -lt \${count} ]]
do
    for idx in \${arr[@]}
	do
        soc_name=Rocket\${idx}Config
        project_name=chipyard.TestHarness.\${soc_name}
        if [[ -e build/\${project_name}-ChipTop/sim-syn-rundir/simv ]]
        then
            c=\`expr \${c} + 1\`
            pass_config[\${#pass_config[*]}]=\${soc_name}
        else
            # check whether we still have this process
            ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
            ret=\$?
            if [[ \${ret} != 0 ]]
            then
                # the process is terminated
                syn2sim \${soc_name}
            fi
        fi
    done
done


# sim.
for idx in \${arr[@]}
do
    soc_name=Rocket\${idx}Config
    project_name=chipyard.TestHarness.\${soc_name}
    sim \${soc_name} \${project_name}
done

# re-check sim.
c=0
pass_config=()
while [[ \${c} -lt \${count} ]]
do
    for idx in \${arr[@]}
	do
        soc_name=Rocket\${idx}Config
        project_name=chipyard.TestHarness.\${soc_name}
        if [[ -e build/\${project_name}-ChipTop/sim-syn-rundir/simv ]]
        then
            c=\`expr \${c} + 1\`
            pass_config[\${#pass_config[*]}]=\${soc_name}
        else
            # check whether we still have this process
            ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
            ret=\$?
            if [[ \${ret} != 0 ]]
            then
                # the process is terminated
                sim \${soc_name} \${project_name}
            fi
        fi
    done
done

# sim2power
for idx in \${arr[@]}
do
    soc_name=Rocket\${idx}Config
    project_name=chipyard.TestHarness.\${soc_name}-ChipTop
    sim2power \${soc_name} \${project_name}
done
echo "[INFO]: Offline Auto-VLSI done."

EOF
}

while getopts "s:e:x:f:" arg
do
    case ${arg} in
        s)
            start=${OPTARG}
            ;;
        e)
            end=${OPTARG}
            ;;
        x)
            run_script=${OPTARG}
            ;;
        f)
            file=${OPTARG}
            ;;
        *)
            echo "[ERROR]: not implemented."
            exit
            ;;
    esac
done

set_env
generate_auto_vlsi_v2
