#!/bin/bash -l
export DATA_DIR=data
declare -a template_paths=($DATA_DIR/20201110_132545_simple_network_K_2_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_3_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_4_exported_output.hdf5
                      $DATA_DIR/20201110_132548_simple_network_K_5_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_1_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_2_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_3_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_4_exported_output.hdf5
                      $DATA_DIR/20201110_132610_simple_network_L_5_exported_output.hdf5
                      $DATA_DIR/20201111_160600_simple_network_N_1_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_2_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_3_exported_output.hdf5
                      $DATA_DIR/20201111_160603_simple_network_N_4_exported_output.hdf5
                      $DATA_DIR/20201111_160602_simple_network_N_5_exported_output.hdf5)
declare -a decode_paths=($DATA_DIR/20210113_121558_simple_network_K_2_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121556_simple_network_K_3_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121848_simple_network_K_4_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121850_simple_network_K_5_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121850_simple_network_L_1_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121850_simple_network_L_2_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121848_simple_network_L_3_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121849_simple_network_L_4_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121850_simple_network_L_5_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_121850_simple_network_N_1_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_122158_simple_network_N_2_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_122158_simple_network_N_3_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_122158_simple_network_N_4_heldout_exported_output.hdf5
                  $DATA_DIR/20210113_122158_simple_network_N_5_heldout_exported_output.hdf5)
arraylength=${#template_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  mpirun -n 5 python -i decode_simple_network_heldout_run.py --plot --interactive --export \
    --template-data-file-path=${template_paths[$i]} --decode-data-file-path=${decode_paths[$i]}
done
