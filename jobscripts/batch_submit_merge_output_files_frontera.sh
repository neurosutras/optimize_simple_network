#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_replay_1 J_replay_2 J_replay_3 J_replay_4 J_replay_5)
declare -a paths=($DATA_DIR/20201021_141037_simple_network_replay_J_reward_1_exported_output.yaml
                  $DATA_DIR/20201021_142106_simple_network_replay_J_reward_2_exported_output.yaml
                  $DATA_DIR/20201021_142412_simple_network_replay_J_reward_3_exported_output.yaml
                  $DATA_DIR/20201021_150420_simple_network_replay_J_reward_4_exported_output.yaml
                  $DATA_DIR/20201021_150419_simple_network_replay_J_reward_5_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
