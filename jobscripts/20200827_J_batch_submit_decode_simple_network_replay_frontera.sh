#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_0 J_1 J_2 J_3 J_4)
declare -a run_paths=($DATA_DIR/20200819_143401_simple_network_J_0_exported_output.hdf5
                      $DATA_DIR/20200819_143255_simple_network_J_1_exported_output.hdf5
                      $DATA_DIR/20200819_143255_simple_network_J_2_exported_output.hdf5
                      $DATA_DIR/20200819_143255_simple_network_J_3_exported_output.hdf5
                      $DATA_DIR/20200819_143255_simple_network_J_4_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20200821_151323_simple_network_replay_J_0_exported_output.hdf5
                  $DATA_DIR/20200821_151334_simple_network_replay_J_1_exported_output.hdf5
                  $DATA_DIR/20200821_151334_simple_network_replay_J_2_exported_output.hdf5
                  $DATA_DIR/20200821_151334_simple_network_replay_J_3_exported_output.hdf5
                  $DATA_DIR/20200821_151337_simple_network_replay_J_4_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
