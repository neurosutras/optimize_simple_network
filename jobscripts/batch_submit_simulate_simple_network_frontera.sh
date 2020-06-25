#!/bin/bash -l
export CONFIG_FILE_PATH="$1"
export LABEL="$2"
export PARAM_FILE_PATH="$3"
export MODEL_KEY="$4"

for ((network_instance=2;network_instance<=5;network_instance++))
do
  export network_instance
  sh simulate_simple_network_frontera.sh $CONFIG_FILE_PATH $LABEL $network_instance $PARAM_FILE_PATH $MODEL_KEY
done
