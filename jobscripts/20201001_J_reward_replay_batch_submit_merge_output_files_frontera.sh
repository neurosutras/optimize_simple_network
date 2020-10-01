#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(J_replay_1 J_replay_2 J_replay_3 J_replay_4 J_replay_5)
declare -a paths=($DATA_DIR/20200930_155409_simple_network_replay_J_1_exported_output.yaml
                  $DATA_DIR/20200930_155409_simple_network_replay_J_2_exported_output.yaml
                  $DATA_DIR/20200930_155701_simple_network_replay_J_3_exported_output.yaml
                  $DATA_DIR/20200930_155702_simple_network_replay_J_4_exported_output.yaml
                  $DATA_DIR/20200930_155702_simple_network_replay_J_5_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
