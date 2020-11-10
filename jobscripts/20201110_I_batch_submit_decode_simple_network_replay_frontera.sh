#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(I_1 I_2 I_3 I_4 I_5)
declare -a run_paths=($DATA_DIR/20201110_132229_simple_network_I_1_exported_output.hdf5
                      $DATA_DIR/20201110_132229_simple_network_I_2_exported_output.hdf5
                      $DATA_DIR/20201110_132229_simple_network_I_3_exported_output.hdf5
                      $DATA_DIR/20201110_132233_simple_network_I_4_exported_output.hdf5
                      $DATA_DIR/20201110_132233_simple_network_I_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201110_132840_simple_network_replay_I_1_exported_output.hdf5
                  $DATA_DIR/20201110_132841_simple_network_replay_I_2_exported_output.hdf5
                  $DATA_DIR/20201110_132841_simple_network_replay_I_3_exported_output.hdf5
                  $DATA_DIR/20201110_132845_simple_network_replay_I_4_exported_output.hdf5
                  $DATA_DIR/20201110_132845_simple_network_replay_I_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
