#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_reward_1 J_reward_2 J_reward_3 J_reward_4 J_reward_5)
declare -a run_paths=($DATA_DIR/20201021_123218_simple_network_J_reward_1_exported_output.hdf5
                      $DATA_DIR/20201021_130515_simple_network_J_reward_2_exported_output.hdf5
                      $DATA_DIR/20201021_130731_simple_network_J_reward_3_exported_output.hdf5
                      $DATA_DIR/20201021_130948_simple_network_J_reward_4_exported_output.hdf5
                      $DATA_DIR/20201021_131311_simple_network_J_reward_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201021_141037_simple_network_replay_J_reward_1_exported_output.hdf5
                         $DATA_DIR/20201021_142106_simple_network_replay_J_reward_2_exported_output.hdf5
                         $DATA_DIR/20201021_142412_simple_network_replay_J_reward_3_exported_output.hdf5
                         $DATA_DIR/20201021_150420_simple_network_replay_J_reward_4_exported_output.hdf5
                         $DATA_DIR/20201021_150419_simple_network_replay_J_reward_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
