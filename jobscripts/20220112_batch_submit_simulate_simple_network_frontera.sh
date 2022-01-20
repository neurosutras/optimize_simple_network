#!/bin/bash -l
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  25866 8 6 config/20220111_J_marder_group_params.yaml 25866
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  26467 8 11 config/20220111_J_marder_group_params.yaml 26467
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  29427 8 16 config/20220111_J_marder_group_params.yaml 29427
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  29623 8 21 config/20220111_J_marder_group_params.yaml 29623
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  no_E_sel 108 1 config/20220112_leave_one_out_best_params.yaml J_no_E_sel
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  no_E_sparse 208 1 config/20220112_leave_one_out_best_params.yaml J_no_E_sparse
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  no_E_rhyth 308 1 config/20220112_leave_one_out_best_params.yaml J_no_E_rhyth
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  no_I_rhyth 408 1 config/20220112_leave_one_out_best_params.yaml J_no_I_rhyth

sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_no_theta 608 1 config/20220112_leave_one_out_best_params.yaml J_no_theta
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_no_gamma 708 1 config/20220112_leave_one_out_best_params.yaml J_no_gamma
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_J_config.yaml \
  J_no_E_sel_or_sparse 508 1 config/20220112_leave_one_out_best_params.yaml J_no_E_sel_or_sparse

sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_K_config.yaml \
  9626 9 6 config/20220111_K_marder_group_params.yaml 9626
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_K_config.yaml \
  12596 9 11 config/20220111_K_marder_group_params.yaml 12596
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_K_config.yaml \
  16954 9 16 config/20220111_K_marder_group_params.yaml 16954
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_K_config.yaml \
  27171 9 21 config/20220111_K_marder_group_params.yaml 27171
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_L_config.yaml \
  7820 11 6 config/20220111_L_marder_group_params.yaml 7820
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_L_config.yaml \
  13211 11 11 config/20220111_L_marder_group_params.yaml 13211
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_L_config.yaml \
  22102 11 16 config/20220111_L_marder_group_params.yaml 22102
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_L_config.yaml \
  23903 11 21 config/20220111_L_marder_group_params.yaml 23903
sh batch_submit_simulate_simple_network_frontera.sh config/simulate_simple_network_L_config.yaml \
  L_27537 11 26 config/20220111_L_marder_group_params.yaml 27537