#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=K
export network_instance_start=16
declare -a run_paths=(20220112_192742_simple_network_16954_9_16_1439807570407388348842912006898658023_exported_output.hdf5
  20220112_192742_simple_network_16954_9_17_1442024374394537463405216424308508265_exported_output.hdf5
  20220112_192742_simple_network_16954_9_18_1446327255900687159810731277153968836_exported_output.hdf5
  20220112_192742_simple_network_16954_9_19_1443475042050173643373636918417818579_exported_output.hdf5
  20220112_192742_simple_network_16954_9_20_1146852724413019391776827063921209527_exported_output.hdf5)
declare -a replay_paths=(20220114_205835_simple_network_replay_K_16954_9_16_10600245364407599256826888698756525952_exported_output.hdf5
  20220114_205601_simple_network_replay_K_16954_9_17_228676875523689654914017283171481758281_exported_output.hdf5
  20220114_205601_simple_network_replay_K_16954_9_18_228672332580851086998034294992490143178_exported_output.hdf5
  20220114_205601_simple_network_replay_K_16954_9_19_228668288775436358945473967922938232353_exported_output.hdf5
  20220114_205600_simple_network_replay_K_16954_9_20_228302593761156018758007594095089491378_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh decode_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done