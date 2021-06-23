#!/bin/bash
# Author: baichen318@gmail.com

# set -ex

benchmarks=(median qsort rsort towers vvadd multiply dhrystone spmv mt-vvadd mt-matmul)

function sims() {
	for bmark in ${benchmarks[@]}
	do
		echo benchmark: ${bmark}
		(set -o pipefail &&  \
			/research/dept8/gds/cbai/research/chipyard/sims/vcs/PATTERN/simv-chipyard-PATTERN \
				+permissive \
				+dramsim \
				+dramsim_ini_dir=/research/dept8/gds/cbai/research/chipyard/generators/testchipip/src/main/resources/dramsim2_ini \
				+max-cycles=10000000  \
				+ntb_random_seed_automatic \
				+verbose \
				+permissive-off \
				/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/${bmark}.riscv </dev/null 2> \
				>(spike-dasm > /research/dept8/gds/cbai/research/chipyard/sims/vcs/output/PATTERN/${bmark}.out) | \
				tee /research/dept8/gds/cbai/research/chipyard/sims/vcs/output/PATTERN/${bmark}.log &)
	done
}

sims
