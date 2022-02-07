#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=M_28922
export network_instance_start=6
declare -a run_paths=(20220203_135832_simple_network_M_28922_12_6_225942413742224729863388731067002127143_exported_output.hdf5
20220203_143921_simple_network_M_28922_12_7_124918937394810990228519210670516639365_exported_output.hdf5
20220203_143921_simple_network_M_28922_12_8_124921561431553462664483409279880748220_exported_output.hdf5
20220203_185200_simple_network_M_28922_12_9_224959193354666550593470576040488923440_exported_output.hdf5
20220203_185549_simple_network_M_28922_12_10_65878464363742190858420090560022507824_exported_output.hdf5)
declare -a replay_paths=(20220203_210045_simple_network_replay_M_28922_12_6_219944034980359200860959331581913239722_exported_output.hdf5
20220203_210620_simple_network_replay_M_28922_12_7_145226950054613402253424147461104650371_exported_output.hdf5
20220203_214333_simple_network_replay_M_28922_12_8_213116379636626219801060680665179823497_exported_output.hdf5
20220203_215130_simple_network_replay_M_28922_12_9_250438542587206607682165199240610651529_exported_output.hdf5
20220203_215324_simple_network_replay_M_28922_12_10_1222444636555413248774622888057377332_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done
