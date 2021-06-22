set -ex
benchmarks=$1
(set -o pipefail &&  \
	/research/dept8/gds/cbai/research/chipyard/sims/vcs/PATTERN/simv-chipyard-PATTERN \
		+permissive \
		+dramsim \
		+dramsim_ini_dir=/research/dept8/gds/cbai/research/chipyard/generators/testchipip/src/main/resources/dramsim2_ini \
		+max-cycles=10000000  \
		+ntb_random_seed_automatic \
		+verbose \
		+permissive-off \
		/research/dept8/gds/cbai/research/chipyard/toolchains/riscv-tools/riscv-tests/build/benchmarks/${benchmarks}.riscv </dev/null 2> \
		>(spike-dasm > /research/dept8/gds/cbai/research/chipyard/sims/vcs/output/PATTERN/${benchmarks}.out) | \
		tee /research/dept8/gds/cbai/research/chipyard/sims/vcs/output/PATTERN/${benchmarks}.log)
