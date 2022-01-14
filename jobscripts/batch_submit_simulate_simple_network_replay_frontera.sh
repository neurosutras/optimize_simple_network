#!/bin/bash -l
export CONFIG_FILE_PATH="$1"
export LABEL="$2"
export network_id="$3"
export network_instance_start="$4"
export PARAM_FILE_PATH="$5"
export MODEL_KEY="$6"

let "network_instance_end = $network_instance_start + 5"

for ((network_instance=$network_instance_start;network_instance<$network_instance_end;network_instance++))
do
  export network_instance
  sh simulate_simple_network_replay_frontera.sh $CONFIG_FILE_PATH $LABEL $network_id $network_instance \
    $PARAM_FILE_PATH $MODEL_KEY
done
