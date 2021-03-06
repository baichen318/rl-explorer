# Author: baichen318@gmail.com

set -ex
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
cd ${SHELL_FOLDER}/..

# python handle-data.py -c configs/area.yml &
python handle-data.py -c configs/latency.yml &
# python handle-data.py -c configs/power.yml &

