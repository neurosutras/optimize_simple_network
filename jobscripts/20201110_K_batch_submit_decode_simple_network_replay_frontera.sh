#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(K_1 K_2 K_3 K_4 K_5)
declare -a run_paths=($DATA_DIR/20201110_132545_simple_network_K_1_exported_output.hdf5
                      $DATA_DIR/20201110_132545_simple_network_K_2_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_3_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_4_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201110_132853_simple_network_replay_K_1_exported_output.hdf5
                  $DATA_DIR/20201110_132921_simple_network_replay_K_2_exported_output.hdf5
                  $DATA_DIR/20201110_132921_simple_network_replay_K_3_exported_output.hdf5
                  $DATA_DIR/20201110_132921_simple_network_replay_K_4_exported_output.hdf5
                  $DATA_DIR/20201110_132921_simple_network_replay_K_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
