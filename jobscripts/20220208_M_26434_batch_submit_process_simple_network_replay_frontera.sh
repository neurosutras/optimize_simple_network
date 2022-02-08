#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=M_26434
export network_instance_start=13
declare -a run_paths=(20220203_190204_simple_network_M_26434_12_13_22943786145013661554817634763241487823_exported_output.hdf5
20220203_190410_simple_network_M_26434_12_14_122988847292883399643226094960716214576_exported_output.hdf5
20220203_192037_simple_network_M_26434_12_15_224247182217749044917088291411980977508_exported_output.hdf5)
declare -a replay_paths=(20220203_223503_simple_network_replay_M_26434_12_13_279229624576042843690987383404782155586_exported_output.hdf5
20220203_224759_simple_network_replay_M_26434_12_14_213458117322223727012772537717337288923_exported_output.hdf5
20220203_225637_simple_network_replay_M_26434_12_15_284045303391991441310485435576822206241_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done
