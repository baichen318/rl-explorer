#Author: baichen318@gmail.com

#!/bin/bash

path="/research/d3/cbai/research/riscv-benchmarks"
sim_path=${sim_path:-"/uac/gds/cbai/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.SmallBoomConfig-ChipTop/sim-syn-rundir/output"}
power_path=${power_path:-"/uac/gds/cbai/cbai/research/synopsys-flow/build/pt-pwr/SmallBoomConfig-benchmarks"}

function set_env() {
    function handler() {
        exit 1
    }

    trap 'handler' SIGINT
}

function help() {
    cat << EOF
    Usage: bash ptpx.sh [-spfrh]
    -s: path to save simulation files
    -p: path to save power reports
    -f: parse failed-list.txt.bak and re-run
    -r: run in parallel
    -h: help
EOF
}

function _ptpx() {
    local bmark=$1
    local _sim_path=$2
    local _bmark=$3

    if [[ ! -e ${_sim_path}/vcdplus.vpd  ]]
    then
        mkdir -p ${_sim_path}

        set -o pipefail && ./simv +permissive +dramsim +max-cycles=2000000 -ucli -do run.tcl \
            +verbose +vcdplusfile=${_sim_path}/vcdplus.vpd \
            +permissive-off ${_bmark} </dev/null 2> \
            >(spike-dasm > ${_sim_path}/${bmark}.chipyard.TestHarness.SmallBoomConfig.out) | \
            tee ${_sim_path}/${bmark}.chipyard.TestHarness.SmallBoomConfig.log
        cat ${_sim_path}/${bmark}.chipyard.TestHarness.SmallBoomConfig.out | grep "PASSED"
        ret=$?

        if [[ $ret == 0 ]] || true 
        then
            pushd ${_sim_path}

            vpd2vcd vcdplus.vpd vcdplus.vcd
            vcd2saif -input vcdplus.vcd -output vcdplus.saif
            cd /research/d3/cbai/research/synopsys-flow/build/pt-pwr
            make build_pt_dir=${power_path}/"build-pt-"${bmark} \
                cur_build_pt_dir=${power_path}/"current-pt-"${bmark} \
                vcs_dir=${_sim_path} \
                icc_dir=/research/d3/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.SmallBoomConfig-ChipTop/syn-rundir
            cd -
            cd ${power_path}
            mv build-pt-${bmark} ${bmark}
            rm -rf current-pt-${bmark}
            cd -
            rm -f ${_sim_path}/*.vcd ${_sim_path}/*.saif

            pushd +1
            popd +1
        fi
    elif [[ -e ${_sim_path}/vcdplus.vpd ]]
    then
        cat ${_sim_path}/${bmark}.chipyard.TestHarness.SmallBoomConfig.out | grep "PASSED"
        ret=$?

        if [[ $ret == 0 ]] || true 
        then
            pushd ${_sim_path}
            # delete redundant files
            rm -f ${_sim_path}/*.vcd ${_sim_path}/*.saif

            vpd2vcd vcdplus.vpd vcdplus.vcd
            vcd2saif -input vcdplus.vcd -output vcdplus.saif
            cd /research/d3/cbai/research/synopsys-flow/build/pt-pwr
            make build_pt_dir=${power_path}/"build-pt-"${bmark} \
                cur_build_pt_dir=${power_path}/"current-pt-"${bmark} \
                vcs_dir=${_sim_path} \
                icc_dir=/research/d3/cbai/research/chipyard/vlsi/build/chipyard.TestHarness.SmallBoomConfig-ChipTop/syn-rundir
            cd -
            cd ${power_path}
            mv build-pt-${bmark} ${bmark}
            rm -rf current-pt-${bmark}
            cd -
            rm -f ${_sim_path}/*.vcd ${_sim_path}/*.saif

            pushd +1
            popd +1
        fi
    fi

    if [[ ! -e ${power_path}/${bmark}/reports/vcdplus.power.avg.max.report ]]
    then
        echo ${bmark} >> logs/failed-list.txt
    fi
}

function ptpx_f() {
    file=$1
    echo Using ${file}...
    cat $file | \
        while read bmark
        do
            _sim_path=${sim_path}/${bmark}
            if [[ ${bmark} == "rv64"* ]] || [[ ${bmark} == "rv32"* ]]
            then
                _bmark=${path}/isa/${bmark}
            else
                _bmark=${path}/benchmarks/${bmark}
            fi
            if [[ -f ${_bmark} ]]
            then
                echo "running:" ${bmark}...
                _ptpx ${bmark} ${_sim_path} ${_bmark} &
            else
                echo $file is wrong
                exit 1
            fi
            sleep 15
        done
}

function ptpx() {
    for bmark in `ls $path`
    do
        _sim_path=${sim_path}/${bmark}
        _bmark=${path}/${bmark}
        _ptpx ${bmark} ${_sim_path} ${_bmark} &
        sleep 300
    done
}

function post() {
    if [[ -f failed-list.txt ]]
    then
        mv -f failed-list.txt failed-list.txt.bak
    fi
}

set_env

while getopts "s:p:f:rh" arg
do
    case $arg in
        f)
            if [[ -f $OPTARG ]]
            then
                ptpx_f ${OPTARG}
                wait
                post
            else
                echo ${OPTARG} not found.
            fi
            ;;
        s)
            if [[ -d $OPTARG ]]
            then
                sim_path=${OPTARG}
            else
                echo ${OPTARG} not found.
            fi
            ;;
        p)
            if [[ -d $OPTARG ]]
            then
                power_path=${OPTARG}
            else
                echo ${OPTARG} not found.
            fi
            ;;
        r)
            ptpx
            wait
            post
            ;;
        h | ?)
            help
            ;;
        ?)
            help
            ;;
    esac
done

echo "done."

