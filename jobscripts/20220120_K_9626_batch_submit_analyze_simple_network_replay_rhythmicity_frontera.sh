#!/bin/bash -l
export DATA_DIR=$SCRATCH/data/optimize_simple_network
export label=K_9626
declare -a replay_paths=(20220114_204109_simple_network_replay_K_9626_9_6_203102570704578757195785856099768050757_exported_output.hdf5
  20220114_204109_simple_network_replay_K_9626_9_8_202945296463734941346958022247753488118_exported_output.hdf5
  20220114_204110_simple_network_replay_K_9626_9_7_203129346654382077969962542300454922825_exported_output.hdf5
  20220114_204905_simple_network_replay_K_9626_9_9_239787384286376395896496898945548353043_exported_output.hdf5
  20220114_204905_simple_network_replay_K_9626_9_10_239789372913255503933585986440391225650_exported_output.hdf5)
arraylength=${#run_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh analyze_simple_network_replay_rhythmicity_frontera.sh $DATA_DIR/${replay_paths[$i]} $label
done
