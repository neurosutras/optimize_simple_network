#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=M_best
export network_instance_start=1
declare -a run_paths=(20220202_180423_simple_network_M_best_12_1_287117176688957895177808778346263217090_exported_output.hdf5
20220202_180614_simple_network_M_best_12_2_35063305012693034409857678449275044802_exported_output.hdf5
20220202_180806_simple_network_M_best_12_3_123182511345430001668208832450472707010_exported_output.hdf5
20220202_181010_simple_network_M_best_12_4_222003998104441445893774456667438580674_exported_output.hdf5
20220202_181209_simple_network_M_best_12_5_315730093555052509487014342439207961538_exported_output.hdf5)
declare -a replay_paths=(20220202_182712_simple_network_replay_M_best_12_1_10222927107912568445557023925682770133_exported_output.hdf5
20220202_183704_simple_network_replay_M_best_12_2_139318673203314862160299532176352553582_exported_output.hdf5
20220202_184350_simple_network_replay_M_best_12_3_120912688664214009580540198742609629623_exported_output.hdf5
20220202_204046_simple_network_replay_M_best_12_4_234992060457651245348997483872935329101_exported_output.hdf5
20220202_213744_simple_network_replay_M_best_12_5_220492997886173062369395880979067569469_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done