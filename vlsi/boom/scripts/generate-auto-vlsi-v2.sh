#!/bin/bash
# Author: baichen318@gmail.com

start=${start:-"1"}
end=${end:-"32"}
sim_script=${sim_script:-""}
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
benchmarks=(median towers mt-vvadd mt-matmul)
function sims2power() {
    soc_name=\${1}
    project_name=\${2}
    for bmark in \${benchmarks[@]}
    do
        echo benchmark: \${bmark}
        (set -o pipefail && \\
            /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/simv \\
                +permissive \\
                +dramsim \\
                +dramsim_ini_dir=/research/dept8/gds/cbai/research/chipyard/generators/testchipip/src/main/resources/dramsim2_ini \\
                +max-cycles=635000  \\
                -ucli -do /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/run.tcl \\
                +ntb_random_seed_automatic \\
                +verbose \\
                +saif /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}/vcdplus.saif \\
                +permissive-off \\
                /research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/\${bmark}.riscv </dev/null 2> \\
                >(spike-dasm > /research/dept8/gds/cbai/research/chipyard/vlsi/buid/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.out) | \\
                tee /research/dept8/gds/cbai/research/chipyard/vlsi/buid/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.log &)
    done

    success_bmark=()
    all_done=0
    while [[ \${all_done} == 0 ]]
    do
        for bmark in \${benchmarks[@]}
        do
            cat /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.out | grep "PASSED"
            ret=\$?
            if [[ \${ret} == 0 ]] && [[ ! \${success_bmark[@]} =~ \${bmark} ]]
            then
                power_name=\${power}/\${soc_name}-power
                mkdir \${power_name}
                cd \${power}
                make build_pt_dir=\${power_name}/"build-pt-"\${bmark} \\
                    cur_build_pt_dir=\${power_name}/"current-pt-"\${bmark} \\
                    vcs_dir=/research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark} \\
                    icc_dir=/research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/syn-rundir/
                mv -f \${power_name}/build-pt-\${bmark} \${power_name}/\${bmark}
                rm -rf \${power_name}/current-pt-\${bmark}
                rm -f /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/*.saif
                cd -
                success_bmark[\${#success_bmark[*]}]=\${bmark}
            fi
        done
        for bmark in \${benchmarks[@]}
        do
            if [[ ! \${success_bmark[@]} =~ \${bmark} ]]
                all_done=0
                break
            fi
            all_done=1
        done
    done
}

# compile
arr=\`seq ${start} ${end}\`
for idx in \${arr[@]}
do
    echo compiling \${idx}-th Config.
    soc_name=Boom\${idx}Config
    make sim-syn \\
        MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
        CONFIG=\${soc_name} \\
        BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
    # 75 sec. would be suitable
    sleep 75
done

# verify all simv have been generated
count=\`expr ${end} - ${start} + 1\`
success_idx=()
c=0
while [[ \${c} -lt \${count} ]]
do
    for idx in \`seq ${start} ${end}\`
    do
        soc_name=Boom\${idx}Config
        project_name=chipyard.TestHarness.\${soc_name}-ChipTop
        if [[ -e build/\${project_name}/sim-syn-rundir/simv ]]
        then
            # simulate
            sims2power \${soc_name} \${project_name}
        else
            if [[ ! \${success_idx[@]} =~ \${soc_name} ]]
            then
                ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
                ret=\$?
                if [[ \${ret} != 0 ]]
                then
                    # no process
                    echo re-compiling \${soc_name}
                    make sim-syn \\
                    MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
                    CONFIG=\${soc_name} BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
                    sleep 60
                fi
            fi
        fi
    done
    # 10 sec. would be suitable
    sleep 10
done

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
            sim_script=${OPTARG}
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
