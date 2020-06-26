#!/bin/bash -l
declare -a labels=(J_2 J_3 J_4 J_reward_0 J_reward_1 J_reward_2 J_reward_3 J_reward_4)
declare -a run_paths=(data/20200625_184737_simple_network_J_2_exported_output.hdf5
                      data/20200625_184737_simple_network_J_3_exported_output.hdf5
                      data/20200625_184737_simple_network_J_4_exported_output.hdf5
                      data/20200625_184837_simple_network_J_reward_0_exported_output.hdf5
                      data/20200625_184837_simple_network_J_reward_1_exported_output.hdf5
                      data/20200625_184837_simple_network_J_reward_2_exported_output.hdf5
                      data/20200625_184943_simple_network_J_reward_3_exported_output.hdf5
                      data/20200625_184943_simple_network_J_reward_4_exported_output.hdf5)
declare -a replay_paths=(data/20200625_184736_simple_network_replay_J_2_exported_output.hdf5
                  data/20200625_184736_simple_network_replay_J_3_exported_output.hdf5
                  data/20200625_184736_simple_network_replay_J_4_exported_output.hdf5
                  data/20200625_184736_simple_network_replay_J_reward_0_exported_output.hdf5
                  data/20200625_184738_simple_network_replay_J_reward_1_exported_output.hdf5
                  data/20200625_184739_simple_network_replay_J_reward_2_exported_output.hdf5
                  data/20200625_184739_simple_network_replay_J_reward_3_exported_output.hdf5
                  data/20200625_184740_simple_network_replay_J_reward_4_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
