#!/bin/bash -l
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_25866 8 6 config/20220111_J_marder_group_params.yaml 25866
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_26467 8 11 config/20220111_J_marder_group_params.yaml 26467
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_29427 8 16 config/20220111_J_marder_group_params.yaml 29427
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_29623 8 21 config/20220111_J_marder_group_params.yaml 29623

sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_K_config.yaml \
  K_9626 9 6 config/20220111_K_marder_group_params.yaml 9626
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_K_config.yaml \
  K_12596 9 11 config/20220111_K_marder_group_params.yaml 12596
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_K_config.yaml \
  K_16954 9 16 config/20220111_K_marder_group_params.yaml 16954
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_K_config.yaml \
  K_27171 9 21 config/20220111_K_marder_group_params.yaml 27171

sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_best 14 1 config/20220128_additional_controls_best_params.yaml O
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_29027 14 6 config/20220203_O_marder_group_params.yaml 29027
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_27288 14 11 config/20220203_O_marder_group_params.yaml 27288
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_18883 14 16 config/20220203_O_marder_group_params.yaml 18883
sh simulate_simple_network_heldout_frontera.sh config/simulate_simple_network_O_config.yaml \
  O_21309 14 21 config/20220203_O_marder_group_params.yaml 21309
