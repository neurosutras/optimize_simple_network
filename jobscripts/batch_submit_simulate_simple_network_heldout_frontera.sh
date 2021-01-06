#!/bin/bash -l
export CONFIG_FILE_PATH="$1"
export LABEL="$2"
export PARAM_FILE_PATH="$3"
export MODEL_KEY="$4"

for ((network_instance=1;network_instance<6;network_instance++))
do
  export network_instance
  sh simulate_simple_network_heldout_frontera.sh $CONFIG_FILE_PATH $LABEL $network_instance $PARAM_FILE_PATH $MODEL_KEY
done
