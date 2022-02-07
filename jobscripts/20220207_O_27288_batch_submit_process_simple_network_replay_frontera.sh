#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=O_27288
export network_instance_start=11
declare -a run_paths=(20220204_212420_simple_network_O_27288_14_11_36385562039890808325056982867046522190_exported_output.hdf5
20220204_212420_simple_network_O_27288_14_12_36516987300432720291449450141038937448_exported_output.hdf5
20220204_212702_simple_network_O_27288_14_13_164958262504219073323785455337049276096_exported_output.hdf5
20220204_212703_simple_network_O_27288_14_14_165633743595338312785681911511763447223_exported_output.hdf5
20220204_212702_simple_network_O_27288_14_15_164977632997672185810211305294293433892_exported_output.hdf5)
declare -a replay_paths=(20220204_234530_simple_network_replay_O_27288_14_11_281614932128676655796846776768847895669_exported_output.hdf5
20220204_234640_simple_network_replay_O_27288_14_12_337069440598384504944173051088697237095_exported_output.hdf5
20220204_234914_simple_network_replay_O_27288_14_13_119030148295250546640953353257123578787_exported_output.hdf5
20220204_234914_simple_network_replay_O_27288_14_14_119026025261673304325537961345955987230_exported_output.hdf5
20220204_234914_simple_network_replay_O_27288_14_15_119034268159701288389656515157049416880_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  let "network_instance = $network_instance_start + $i"
  sh process_simple_network_replay_frontera.sh $DATA_DIR/${run_paths[$i]} $DATA_DIR/${replay_paths[$i]} \
    "$label"_"$network_instance"
done
