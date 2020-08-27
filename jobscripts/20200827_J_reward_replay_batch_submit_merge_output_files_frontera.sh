#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_reward_replay_0 J_reward_replay_1 J_reward_replay_2 J_reward_replay_3 J_reward_replay_4)
declare -a paths=($DATA_DIR/20200821_151349_simple_network_replay_J_reward_0_exported_output.yaml
                  $DATA_DIR/20200821_151907_simple_network_replay_J_reward_1_exported_output.yaml
                  $DATA_DIR/20200821_151950_simple_network_replay_J_reward_2_exported_output.yaml
                  $DATA_DIR/20200821_151952_simple_network_replay_J_reward_3_exported_output.yaml
                  $DATA_DIR/20200821_151951_simple_network_replay_J_reward_4_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
