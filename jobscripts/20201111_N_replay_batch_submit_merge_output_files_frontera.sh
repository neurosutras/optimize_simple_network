#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(N_replay_1 N_replay_2 N_replay_3 N_replay_4 N_replay_5)
declare -a paths=($DATA_DIR/20201111_160645_simple_network_replay_N_1_exported_output.yaml
                  $DATA_DIR/20201111_160645_simple_network_replay_N_2_exported_output.yaml
                  $DATA_DIR/20201111_160645_simple_network_replay_N_3_exported_output.yaml
                  $DATA_DIR/20201111_160647_simple_network_replay_N_4_exported_output.yaml
                  $DATA_DIR/20201111_160647_simple_network_replay_N_5_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
