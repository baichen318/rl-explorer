# Author: baichen318@gmail.com
# set -ex

start=${start:-"1"}
end=${end:-"10"}
name=${name:-"DefaultConfig"}
file=${file:-"compile-default.sh"}

function set_env() {
    function handler() {
        exit 1
    }

    trap 'handler' SIGINT
}

function generate_compilation() {
    script="arr=`seq ${start} ${end}`
        for idx in ${arr[@]}
        do
            echo running ${idx}
            make sim-syn \
            MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json -hir chipyard.TestHarness.PATTERN.hir'
            CONFIG=PATTERN BINARY=/research/dept8/gds/cbai/demo/chipyard/riscv-tools-install/riscv64-unknown-elf/share/riscv-tests/benchmarks/dhrystone.riscv
            sleep 60
        done
    "
    sed -i 's/PATTERN/%s/g' ${name} ${script}
    echo ${script} > ${file}
}

while getopts "s:e:n:" arg
do
    case $arg in
        s)
            start=${OPTARG}
            ;;
        e)
            end=${OPTARG}
            ;;
        n)
            name=${OPTARG}
            ;;
        *)
            echo "Undefined parameters"
            exit
            ;;
    esac
done

generate_compilation
