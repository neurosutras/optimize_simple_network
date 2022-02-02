#!/bin/bash -l
sh batch_submit_simulate_simple_network_replay_frontera.sh config/simulate_simple_network_replay_M_config.yaml \
  M_best 12 1 config/20220128_additional_controls_best_params.yaml M
sh batch_submit_simulate_simple_network_replay_frontera.sh config/simulate_simple_network_replay_M_config.yaml \
  M_28922 12 6 config/20220202_M_marder_group_params.yaml 28922
sh batch_submit_simulate_simple_network_replay_frontera.sh config/simulate_simple_network_replay_M_config.yaml \
  M_26434 12 11 config/20220202_M_marder_group_params.yaml 26434
sh batch_submit_simulate_simple_network_replay_frontera.sh config/simulate_simple_network_replay_M_config.yaml \
  M_29613 12 16 config/20220202_M_marder_group_params.yaml 29613
sh batch_submit_simulate_simple_network_replay_frontera.sh config/simulate_simple_network_replay_M_config.yaml \
  M_28842 12 21 config/20220202_M_marder_group_params.yaml 28842
