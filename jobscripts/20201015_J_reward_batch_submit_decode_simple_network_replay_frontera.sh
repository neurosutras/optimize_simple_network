#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_reward_1 J_reward_2 J_reward_3 J_reward_4 J_reward_5)
declare -a run_paths=($DATA_DIR/20201001_103734_simple_network_J_reward_1_exported_output.hdf5
                      $DATA_DIR/20201001_111057_simple_network_J_reward_2_exported_output.hdf5
                      $DATA_DIR/20201001_111210_simple_network_J_reward_3_exported_output.hdf5
                      $DATA_DIR/20201001_111358_simple_network_J_reward_4_exported_output.hdf5
                      $DATA_DIR/20201001_111510_simple_network_J_reward_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201001_104300_simple_network_replay_J_reward_1_exported_output.hdf5
                         $DATA_DIR/20201001_104738_simple_network_replay_J_reward_2_exported_output.hdf5
                         $DATA_DIR/20201001_105219_simple_network_replay_J_reward_3_exported_output.hdf5
                         $DATA_DIR/20201001_105738_simple_network_replay_J_reward_4_exported_output.hdf5
                         $DATA_DIR/20201001_111402_simple_network_replay_J_reward_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
