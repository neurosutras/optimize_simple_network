#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=J
export network_instance_start=11
declare -a run_paths=(20220112_162748_simple_network_26467_8_11_296957133669439240736160398864100151630_exported_output.hdf5
  20220112_163407_simple_network_26467_8_12_256313855129711882895004287113154503513_exported_output.hdf5
  20220112_163408_simple_network_26467_8_13_257814232911354387790800071906758068108_exported_output.hdf5
  20220112_163408_simple_network_26467_8_14_257818630866655554604418298836715109900_exported_output.hdf5
  20220112_163408_simple_network_26467_8_15_257782642266115075171279713668300602694_exported_output.hdf5)
declare -a replay_paths=(20220114_114204_simple_network_replay_J_26467_8_11_97572775211990371227734214715126845838_exported_output.hdf5
  20220114_114204_simple_network_replay_J_26467_8_12_97756727951997615309483345787749076801_exported_output.hdf5
  20220114_114204_simple_network_replay_J_26467_8_13_97750633721737018098619204547969231453_exported_output.hdf5
  20220114_114157_simple_network_replay_J_26467_8_14_92377531065810012967525081211963383206_exported_output.hdf5
  20220114_192128_simple_network_replay_J_26467_8_15_157798003209251242948535320763897147700_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh decode_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done