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

function generate_compilation() {
    script="arr=\`seq ${start} ${end}\` \n
		for idx in \${arr[@]} \n
		do \n
            echo running \${idx} \n
            make sim-syn \ \n
			MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json -hir chipyard.TestHarness.BOOM${model}Config\${idx}Config.hir' \ \n
			CONFIG=BOOM${model}Config\${idx}Config BINARY=/research/dept8/gds/cbai/demo/chipyard/riscv-tools-install/riscv64-unknown-elf/share/riscv-tests/benchmarks/dhrystone.riscv \n
			sleep 60 \n
		done \n
    "
    echo "generating compile script: " ${file}
    echo -e ${script} > ${file}
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
generate_compilation
