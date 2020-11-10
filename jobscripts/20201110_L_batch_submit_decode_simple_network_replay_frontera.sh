#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(L_1 L_2 L_3 L_4 L_5)
declare -a run_paths=($DATA_DIR/20201110_132610_simple_network_L_1_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_2_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_3_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_4_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201110_132936_simple_network_replay_L_1_exported_output.hdf5
                  $DATA_DIR/20201110_132936_simple_network_replay_L_2_exported_output.hdf5
                  $DATA_DIR/20201110_132940_simple_network_replay_L_3_exported_output.hdf5
                  $DATA_DIR/20201110_132939_simple_network_replay_L_4_exported_output.hdf5
                  $DATA_DIR/20201110_132939_simple_network_replay_L_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
