# Author: baichen318@gmail.com
set -ex
make sim-syn \
  MACROCOMPILER_MODE='-l /research/dept8/gds/cbai/research/chipyard/vlsi/hammer/src/hammer-vlsi/technology/asap7/sram-cache.json -hir chipyard.TestHarness.PATTERN.hir' \
  CONFIG=PATTERN BINARY=/research/dept8/gds/cbai/demo/chipyard/riscv-tools-install/riscv64-unknown-elf/share/riscv-tests/benchmarks/dhrystone.riscv
