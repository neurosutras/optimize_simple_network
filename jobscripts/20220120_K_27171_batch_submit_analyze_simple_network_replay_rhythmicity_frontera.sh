#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=K_27171
declare -a replay_paths=(20220114_210347_simple_network_replay_K_27171_9_21_258110763324854952858463988216012410897_exported_output.hdf5
  20220114_210347_simple_network_replay_K_27171_9_22_258095466743518323840054792163057933803_exported_output.hdf5
  20220114_210347_simple_network_replay_K_27171_9_23_258107712248316528538951894902841744842_exported_output.hdf5
  20220114_210347_simple_network_replay_K_27171_9_24_258099808446824105528198965594315290544_exported_output.hdf5
  20220114_210347_simple_network_replay_K_27171_9_25_258096554546189644693057867219666208380_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh analyze_simple_network_replay_rhythmicity_frontera.sh $DATA_DIR/${replay_paths[$i]} $label
done
