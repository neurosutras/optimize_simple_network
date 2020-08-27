#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_reward_0 J_reward_1 J_reward_2 J_reward_3 J_reward_4)
declare -a run_paths=($DATA_DIR/20200821_151542_simple_network_J_reward_0_exported_output.hdf5
                      $DATA_DIR/20200821_151901_simple_network_J_reward_1_exported_output.hdf5
                      $DATA_DIR/20200821_151956_simple_network_J_reward_2_exported_output.hdf5
                      $DATA_DIR/20200821_151955_simple_network_J_reward_3_exported_output.hdf5
                      $DATA_DIR/20200821_151956_simple_network_J_reward_4_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20200821_151349_simple_network_replay_J_reward_0_exported_output.hdf5
                  $DATA_DIR/20200821_151907_simple_network_replay_J_reward_1_exported_output.hdf5
                  $DATA_DIR/20200821_151950_simple_network_replay_J_reward_2_exported_output.hdf5
                  $DATA_DIR/20200821_151952_simple_network_replay_J_reward_3_exported_output.hdf5
                  $DATA_DIR/20200821_151951_simple_network_replay_J_reward_4_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
