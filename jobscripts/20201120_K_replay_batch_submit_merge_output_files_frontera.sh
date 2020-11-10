#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(K_replay_1 K_replay_2 K_replay_3 K_replay_4 K_replay_5)
declare -a paths=($DATA_DIR/20201110_132853_simple_network_replay_K_1_exported_output.yaml
                  $DATA_DIR/20201110_132921_simple_network_replay_K_2_exported_output.yaml
                  $DATA_DIR/20201110_132921_simple_network_replay_K_3_exported_output.yaml
                  $DATA_DIR/20201110_132921_simple_network_replay_K_4_exported_output.yaml
                  $DATA_DIR/20201110_132921_simple_network_replay_K_5_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
