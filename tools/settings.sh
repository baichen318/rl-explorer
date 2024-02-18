#!/bin/bash
# Author: baichen318@gmail.com


function set_handler() {
	function handler() {
		exit 1;
	}
	trap 'handler' SIGINT
}


function set_env_variables() {
	echo "[INFO]: set up environmental variables..."
	ROOT=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd);
	export ROOT
	export PYTHONPATH="${ROOT}"
}


function setup() {
	set_handler
	set_env_variables
}

setup
echo "[INFO]: set up done."
