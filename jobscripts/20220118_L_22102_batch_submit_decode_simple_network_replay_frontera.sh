#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=L
export network_instance_start=16
declare -a run_paths=(20220112_193713_simple_network_22102_11_16_113171518182060689372077296585542425773_exported_output.hdf5
  20220112_193713_simple_network_22102_11_17_113017911412859658814753205745500356195_exported_output.hdf5
  20220112_193712_simple_network_22102_11_18_112887629414702827682975704442219473639_exported_output.hdf5
  20220112_193713_simple_network_22102_11_19_113165723434254396078415008731899889257_exported_output.hdf5
  20220112_193713_simple_network_22102_11_20_113168603377961789587653143182613127876_exported_output.hdf5)
declare -a replay_paths=(20220114_211146_simple_network_replay_L_22102_11_16_296649315462235215972199784258883789682_exported_output.hdf5
  20220114_211146_simple_network_replay_L_22102_11_17_296653293508275057184559112378215259241_exported_output.hdf5
  20220114_211146_simple_network_replay_L_22102_11_18_296654583342760789405748105257991991557_exported_output.hdf5
  20220114_211511_simple_network_replay_L_22102_11_19_119450779003090903492019379834329038355_exported_output.hdf5
  20220114_211511_simple_network_replay_L_22102_11_20_119456283775822394580107157309817024384_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh decode_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done