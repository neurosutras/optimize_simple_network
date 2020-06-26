#!/bin/bash -l
declare -a labels=(J_2 J_3 J_4 J_replay_0 J_replay_1 J_replay_2 J_replay_3 J_replay_4)
declare -a paths=(data/20200625_184736_simple_network_replay_J_2_exported_output.yaml
                  data/20200625_184736_simple_network_replay_J_3_exported_output.yaml
                  data/20200625_184736_simple_network_replay_J_4_exported_output.yaml
                  data/20200625_184736_simple_network_replay_J_reward_0_exported_output.yaml
                  data/20200625_184738_simple_network_replay_J_reward_1_exported_output.yaml
                  data/20200625_184739_simple_network_replay_J_reward_2_exported_output.yaml
                  data/20200625_184739_simple_network_replay_J_reward_3_exported_output.yaml
                  data/20200625_184740_simple_network_replay_J_reward_4_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
