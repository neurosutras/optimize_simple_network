#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=J
export network_instance_start=0
declare -a run_paths=(20200819_143401_simple_network_J_0_exported_output.hdf5
                      20200819_143255_simple_network_J_1_exported_output.hdf5
                      20200819_143255_simple_network_J_2_exported_output.hdf5
                      20200819_143255_simple_network_J_3_exported_output.hdf5
                      20200819_143255_simple_network_J_4_exported_output.hdf5)
declare -a replay_paths=(20200819_143207_simple_network_replay_J_0_exported_output.hdf5
                  20200819_143207_simple_network_replay_J_1_exported_output.hdf5
                  20200819_143207_simple_network_replay_J_2_exported_output.hdf5
                  20200819_143207_simple_network_replay_J_3_exported_output.hdf5
                  20200819_143207_simple_network_replay_J_4_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh decode_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done
