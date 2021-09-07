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
benchmarks=(median towers mt-vvadd mt-matmul)
success_idx=()
function sims2power() {
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
		sleep 35
    done

    success_bmark=()
    all_done=0
    while [[ \${all_done} == 0 ]]
    do
        for bmark in \${benchmarks[@]}
        do
            ret=\`grep "PASSED" /research/dept8/gds/cbai/research/chipyard/vlsi/build/\${project_name}/sim-syn-rundir/\${bmark}.riscv/\${bmark}.out\`
            if [[ ! -z \${ret} ]] && [[ ! \${success_bmark[@]} =~ \${bmark} ]]
            then
				# 10s to generate saif
				sleep 10
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
                success_bmark[\${#success_bmark[*]}]=\${bmark}
            fi
        done
		c=0
        for bmark in \${benchmarks[@]}
        do
            if [[ ! \${success_bmark[@]} =~ \${bmark} ]]
			then
                all_done=0
			else
				c=\`expr \${c} + 1\`
            fi
			if [[ \${c} == \${#benchmarks[*]} ]]
			then
            	all_done=1
				success_idx[\${success_idx[*]}]=\${soc_name}
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

# compile
arr=\`seq ${start} ${end}\`
# for idx in \${arr[@]}
# do
#     echo compiling \${idx}-th Config.
#     soc_name=Rocket\${idx}Config
#     make sim-syn \\
#         MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
#         CONFIG=\${soc_name} \\
#         BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
#     # 75 sec. would be suitable
#     sleep 75
# done

# verify all simv have been generated
count=\`expr ${end} - ${start} + 1\`
running_idx=()
while :
do
    for idx in \`seq ${start} ${end}\`
    do
        soc_name=Rocket\${idx}Config
        project_name=chipyard.TestHarness.\${soc_name}-ChipTop
        if [[ -e build/\${project_name}/sim-syn-rundir/simv ]] && \\
			[[ ! \${success_idx[@]} =~ \${soc_name} ]] && \\
			[[ ! \${running_idx[@]} =~ \${soc_name} ]]
        then
            # simulate & power
			running_idx[\${running_idx[*]}]=\${soc_name}
            sims2power \${soc_name} \${project_name} &
        # else
        #     if [[ ! \${success_idx[@]} =~ \${soc_name} ]]
        #     then
        #         ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
        #         ret=\$?
        #         if [[ \${ret} != 0 ]]
        #         then
        #             # no process
        #             echo re-compiling \${soc_name}
        #             make sim-syn \\
        #             MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' \\
        #             CONFIG=\${soc_name} BINARY=/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/towers.riscv &
        #             sleep 60
        #         fi
        #     fi
        fi
    done
    # 10 sec. would be suitable
    sleep 10
	success=()
	sum=0
	for idx in \`seq ${start} ${end}\`
	do
		soc_name=Rocket\${idx}Config
		check_power_report \${soc_name}
		success[\${#success[*]}]=\$?
	done
	for ((i=0; i < \${#success[*]}; i++))
	do
		sum=\`expr \${sum} + \${success[\$i]}\`
	done
	if [[ \${sum} == \${count} ]]
	then
		echo "[INFO]: Offline Auto-VLSI flow done."
		break
	else
		echo "[INFO]: Offline Auto-VLSI flow continue..."
	fi
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
