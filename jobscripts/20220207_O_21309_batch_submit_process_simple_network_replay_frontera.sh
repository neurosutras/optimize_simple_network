#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=O_21309
export network_instance_start=21
declare -a run_paths=(20220204_221442_simple_network_O_21309_14_21_49083015017612386138160607662313408273_exported_output.hdf5
20220204_221442_simple_network_O_21309_14_22_49060068957185004902384356324612644037_exported_output.hdf5
20220204_221653_simple_network_O_21309_14_23_152508199147052560807198173029498530224_exported_output.hdf5
20220204_221653_simple_network_O_21309_14_24_152690332808448477378231687859156125457_exported_output.hdf5
20220204_221653_simple_network_O_21309_14_25_152695357458515132022816293242125819077_exported_output.hdf5)
declare -a replay_paths=(20220204_235657_simple_network_replay_O_21309_14_21_145729338210233236155222486089489652912_exported_output.hdf5
20220204_235808_simple_network_replay_O_21309_14_22_201807205147505440542103485283857997070_exported_output.hdf5
20220204_235808_simple_network_replay_O_21309_14_23_201800664070408262878740228316486173848_exported_output.hdf5
20220205_000255_simple_network_replay_O_21309_14_24_89050729593581857410781475771767463139_exported_output.hdf5
20220205_000239_simple_network_replay_O_21309_14_25_76221918540808642201932220949024403723_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done
