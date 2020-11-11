#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
declare -a labels=(N_1 N_2 N_3 N_4 N_5)
declare -a run_paths=($DATA_DIR/20201111_160600_simple_network_N_1_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_2_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_3_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_4_exported_output.hdf5
                      $DATA_DIR/20201111_160602_simple_network_N_5_exported_output.hdf5)
declare -a replay_paths=($DATA_DIR/20201111_160645_simple_network_replay_N_1_exported_output.hdf5
                  $DATA_DIR/20201111_160645_simple_network_replay_N_2_exported_output.hdf5
                  $DATA_DIR/20201111_160645_simple_network_replay_N_3_exported_output.hdf5
                  $DATA_DIR/20201111_160647_simple_network_replay_N_4_exported_output.hdf5
                  $DATA_DIR/20201111_160647_simple_network_replay_N_5_exported_output.hdf5)
arraylength=${#labels[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh decode_simple_network_replay_frontera.sh ${run_paths[$i]} ${replay_paths[$i]} ${labels[$i]}
done
