#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=P_15131
export network_instance_start=21
declare -a run_paths=(20220204_223221_simple_network_P_15131_15_21_207504866921895572593845399055128992965_exported_output.hdf5
20220204_223433_simple_network_P_15131_15_22_311754841576121243580836594785725622704_exported_output.hdf5
20220204_223433_simple_network_P_15131_15_23_311973196768855431522217514798353843985_exported_output.hdf5
20220204_223433_simple_network_P_15131_15_24_311977991657250794798638394784986569925_exported_output.hdf5
20220204_223643_simple_network_P_15131_15_25_74722211980144451046944190846968179120_exported_output.hdf5)
declare -a replay_paths=(20220205_001102_simple_network_replay_P_15131_15_21_134602183927516107216621159223007255460_exported_output.hdf5
20220205_001043_simple_network_replay_P_15131_15_22_119837832825927539955565437231353781542_exported_output.hdf5
20220205_001102_simple_network_replay_P_15131_15_23_134601507319008235398664137050197566469_exported_output.hdf5
20220205_001102_simple_network_replay_P_15131_15_24_134608261519862576434430556465334648211_exported_output.hdf5
20220205_001102_simple_network_replay_P_15131_15_25_134446838100147888558253885346697446821_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done