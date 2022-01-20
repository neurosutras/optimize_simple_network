#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=L_7820
declare -a replay_paths=(20220114_210347_simple_network_replay_L_7820_11_6_258114603513892019248416093747396212724_exported_output.hdf5
  20220114_210347_simple_network_replay_L_7820_11_7_258112999935882730541887723139637243194_exported_output.hdf5
  20220114_210347_simple_network_replay_L_7820_11_8_258105896338831701600550986607567265126_exported_output.hdf5
  20220114_210347_simple_network_replay_L_7820_11_9_258108495814843794611867808491931103493_exported_output.hdf5
  20220114_210347_simple_network_replay_L_7820_11_10_258115597035049948126778900968200933398_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh analyze_simple_network_replay_rhythmicity_frontera.sh $DATA_DIR/${replay_paths[$i]} $label
done
