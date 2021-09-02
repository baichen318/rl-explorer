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

function generate_auto_vlsi() {
    echo "generating compilation script: " ${file}
cat > ${file} << EOF
#!/bin/bash
# Author: baichen318@gmail.com
# Auto-generated by ${BASH_SOURCE[0]}

# compile
arr=\`seq ${start} ${end}\`
for idx in \${arr[@]}
do
    echo compiling \${idx}-th Config.
    soc_name=Boom\${idx}Config
    make -j80 MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' CONFIG=\${soc_name} &
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
        if [[ -e simv-chipyard-\${soc_name} ]]
        then
            mkdir -p \${soc_name}
            mkdir -p output/\${soc_name}
            chmod +x simv-chipyard-\${soc_name}
            mv -f simv-chipyard-\${soc_name}* \${soc_name}
            # sim. script
            cp -f ${sim_script} \${soc_name}
            sed -i "s/PATTERN/\${soc_name}/g" \${soc_name}/sim.sh
            c=\`expr \${c} + 1\`
			success_idx[\${#success_idx[*]}]=\${soc_name}
            # simulate
            cd \${soc_name}
            bash sim.sh
            cd -
            sleep 15
		else
			if [[ ! \${success_idx[@]} =~ \${soc_name} ]]
			then
				ps aux | grep cbai | grep \${soc_name} | grep -v grep > /dev/null
				ret=\$?
				if [[ \${ret} != 0 ]]
				then
					# no process
					echo re-compiling \${soc_name}
					make -j 80 MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json' CONFIG=\${soc_name} &
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
generate_auto_vlsi
