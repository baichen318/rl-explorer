# Author: baichen318@gmail.com

set -ex
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
cd ${SHELL_FOLDER}/..

python sample.py -c configs/design-explorer.yml

