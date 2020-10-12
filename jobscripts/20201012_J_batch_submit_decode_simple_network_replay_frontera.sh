#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_1 J_2 J_3 J_4 J_5)
declare -a run_paths=($DATA_DIR/20200930_154539_simple_network_J_1_exported_output.hdf5
                      $DATA_DIR/20200930_155406_simple_network_J_2_exported_output.hdf5
                      $DATA_DIR/20200930_155406_simple_network_J_3_exported_output.hdf5
                      $DATA_DIR/20200930_155657_simple_network_J_4_exported_output.hdf5
                      $DATA_DIR/20200930_155657_simple_network_J_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20200930_155409_simple_network_replay_J_1_exported_output.hdf5
                  $DATA_DIR/20200930_155409_simple_network_replay_J_2_exported_output.hdf5
                  $DATA_DIR/20200930_155701_simple_network_replay_J_3_exported_output.hdf5
                  $DATA_DIR/20200930_155702_simple_network_replay_J_4_exported_output.hdf5
                  $DATA_DIR/20200930_155702_simple_network_replay_J_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
