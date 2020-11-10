#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(L_replay_1 L_replay_2 L_replay_3 L_replay_4 L_replay_5)
declare -a paths=($DATA_DIR/20201110_132936_simple_network_replay_L_1_exported_output.yaml
                  $DATA_DIR/20201110_132936_simple_network_replay_L_2_exported_output.yaml
                  $DATA_DIR/20201110_132940_simple_network_replay_L_3_exported_output.yaml
                  $DATA_DIR/20201110_132939_simple_network_replay_L_4_exported_output.yaml
                  $DATA_DIR/20201110_132939_simple_network_replay_L_5_exported_output.yaml)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh merge_output_files_frontera.sh ${paths[$i]} ${labels[$i]}
done
