#!/bin/bash -l
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_best 14 1 config/20220128_additional_controls_best_params.yaml O
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_29027 14 6 config/20220203_O_marder_group_params.yaml 29027
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_27288 14 11 config/20220203_O_marder_group_params.yaml 27288
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_18883 14 16 config/20220203_O_marder_group_params.yaml 18883
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_21309 14 21 config/20220203_O_marder_group_params.yaml 21309
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_P_config.yaml \
  P_best 15 1 config/20220128_additional_controls_best_params.yaml P
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_P_config.yaml \
  P_24612 15 6 config/20220203_P_marder_group_params.yaml 24612
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_P_config.yaml \
  P_22746 15 11 config/20220203_P_marder_group_params.yaml 22746
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_P_config.yaml \
  P_18307 15 16 config/20220203_P_marder_group_params.yaml 18307
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_P_config.yaml \
  P_15131 15 21 config/20220203_P_marder_group_params.yaml 15131