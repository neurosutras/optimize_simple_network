from simple_network_analysis_utils import *
from scipy.stats import kstest
from scipy.stats import mannwhitneyu, wilcoxon, ttest_rel, ttest_ind, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection


export_data_file_path = 'data/20211025_simple_network_instances_summary_data_for_stats.hdf5'

run_band_filter_range = {'Theta': (4., 10.), 'Gamma': (30., 100.)}
replay_band_filter_range = {'Ripple': (75., 300.)}

psd_area_instance_dict = defaultdict(
    lambda: {'Theta': defaultdict(list), 'Gamma': defaultdict(list), 'Ripple': defaultdict(list)})
nested_gamma_psd_area_instance_dict = defaultdict(lambda: {'Theta': defaultdict(list)})

run_decoded_pos_error_instance_dict = defaultdict(lambda: defaultdict(list))
theta_sequence_score_instances_dict = defaultdict(lambda: defaultdict(list))

cdf_bins = 100
cdf_p = np.arange(1., cdf_bins + 1.) / cdf_bins

replay_decoded_pos_instance_dict = defaultdict(lambda: defaultdict(list))
replay_decoded_path_len_instance_dict = defaultdict(lambda: defaultdict(list))
replay_decoded_velocity_instance_dict = defaultdict(lambda: defaultdict(list))
offline_sequence_fraction_instances_dict = defaultdict(dict)

spatial_mod_depth_instances_dict = defaultdict(lambda: {'predicted': {}, 'actual': {}})
delta_peak_locs_instances_dict = defaultdict(dict)

with h5py.File(export_data_file_path, 'r') as f:
    fft_f = f['shared_context']['fft_f'][:]
    replay_fft_f = f['shared_context']['replay_fft_f'][:]
    fft_f_nested_gamma = f['shared_context']['fft_f_nested_gamma'][:]
    for model_key in f:
        if model_key == 'shared_context':
            continue
        if 'fft_power' in f[model_key]:
            for pop_name in f[model_key]['fft_power']:
                for instance in f[model_key]['fft_power'][pop_name][:]:
                    d_fft_f = fft_f[1] - fft_f[0]
                    for band, this_band_filter_range in run_band_filter_range.items():
                        indexes = np.where((fft_f >= this_band_filter_range[0]) & (fft_f <= this_band_filter_range[1]))
                        this_area = np.trapz(instance[indexes], dx=d_fft_f)
                        psd_area_instance_dict[model_key][band][pop_name].append(this_area)
        if 'fft_power_nested_gamma' in f[model_key]:
            for pop_name in f[model_key]['fft_power_nested_gamma']:
                for instance in f[model_key]['fft_power_nested_gamma'][pop_name][:]:
                    d_fft_f_nested_gamma = fft_f_nested_gamma[1] - fft_f_nested_gamma[0]
                    band = 'Theta'
                    this_band_filter_range = run_band_filter_range[band]
                    indexes = np.where((fft_f_nested_gamma >= this_band_filter_range[0]) & (
                                fft_f_nested_gamma <= this_band_filter_range[1]))
                    this_area = np.trapz(instance[indexes], dx=d_fft_f_nested_gamma)
                    nested_gamma_psd_area_instance_dict[model_key][band][pop_name].append(this_area)

        if 'decoded_pos_error' in f[model_key]:
            for pop_name in f[model_key]['decoded_pos_error']:
                for instance in f[model_key]['decoded_pos_error'][pop_name].values():
                    this_mean_error = np.mean(instance[:])
                    run_decoded_pos_error_instance_dict[model_key][pop_name].append(this_mean_error)
        if 'theta_sequence_score' in f[model_key]:
            for pop_name in f[model_key]['theta_sequence_score']:
                for instance in f[model_key]['theta_sequence_score'][pop_name].values():
                    this_mean_seq_score = np.mean(instance[:])
                    theta_sequence_score_instances_dict[model_key][pop_name].append(this_mean_seq_score)

        if 'spatial_modulation_depth' in f[model_key]:
            for condition in ['predicted', 'actual']:
                for pop_name in f[model_key]['spatial_modulation_depth'][condition]:
                    spatial_mod_depth_instances_dict[model_key][condition][pop_name] = \
                        f[model_key]['spatial_modulation_depth'][condition][pop_name][:]

        if 'delta_peak_locs' in f[model_key]:
            for pop_name in f[model_key]['delta_peak_locs']:
                delta_peak_locs_instances_dict[model_key][pop_name] = f[model_key]['delta_peak_locs'][pop_name][:]

        if 'replay_fft_power' in f[model_key]:
            for pop_name in f[model_key]['replay_fft_power']:
                for instance in f[model_key]['replay_fft_power'][pop_name][:]:
                    d_replay_fft_f = replay_fft_f[1] - replay_fft_f[0]
                    band = 'Ripple'
                    this_band_filter_range = replay_band_filter_range[band]
                    indexes = np.where(
                        (replay_fft_f >= this_band_filter_range[0]) & (replay_fft_f <= this_band_filter_range[1]))
                    this_area = np.trapz(instance[indexes], dx=d_replay_fft_f)
                    psd_area_instance_dict[model_key][band][pop_name].append(this_area)

        if 'replay_decoded_pos' in f[model_key]:
            for pop_name in f[model_key]['replay_decoded_pos']:
                for instance in f[model_key]['replay_decoded_pos'][pop_name].values():
                    a = np.sort(instance[:])
                    q = [np.quantile(a, pi) for pi in cdf_p]
                    replay_decoded_pos_instance_dict[model_key][pop_name].extend(q)

        if 'replay_decoded_velocity' in f[model_key]:
            for pop_name in f[model_key]['replay_decoded_velocity']:
                for instance in f[model_key]['replay_decoded_velocity'][pop_name].values():
                    a = np.sort(instance[:])
                    q = [np.quantile(a, pi) for pi in cdf_p]
                    replay_decoded_velocity_instance_dict[model_key][pop_name].extend(q)

        if 'replay_decoded_path_len' in f[model_key]:
            for pop_name in f[model_key]['replay_decoded_path_len']:
                for instance in f[model_key]['replay_decoded_path_len'][pop_name].values():
                    a = np.sort(instance[:])
                    q = [np.quantile(a, pi) for pi in cdf_p]
                    replay_decoded_path_len_instance_dict[model_key][pop_name].extend(q)

        if 'replay_sequence_fraction' in f[model_key]:
            for pop_name in f[model_key]['replay_sequence_fraction']:
                offline_sequence_fraction_instances_dict[model_key][pop_name] = \
                    f[model_key]['replay_sequence_fraction'][pop_name][:]

ordered_model_key_list = ['J', 'K', 'O_best', 'N', 'M_best', 'P_best', 'L', 'J_reward']
control_model_key = 'J'

for band in psd_area_instance_dict[control_model_key]:
    within_count = 0
    within_label_list, within_stat_list, within_p_val_list = [], [], []
    for model_key in ordered_model_key_list:
        if model_key not in psd_area_instance_dict:
            continue
        for control_pop_name in ['FF']:
            for pop_name in ['E', 'I']:
                within_count += 1
                this_label = '%s; %s < %s' % (model_key, pop_name, control_pop_name)
                this_stat, this_p_val = ttest_rel(psd_area_instance_dict[model_key][band][pop_name],
                                                  psd_area_instance_dict[model_key][band][control_pop_name], \
                                                  alternative='greater')
                within_stat_list.append(this_stat)
                within_p_val_list.append(this_p_val)
                within_label_list.append(this_label)
    across_label_list, across_stat_list, across_p_val_list = [], [], []
    for model_key in ordered_model_key_list:
        if model_key not in psd_area_instance_dict:
            continue
        if model_key != control_model_key:
            for pop_name in ['E', 'I']:
                this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
                this_stat, this_p_val = ttest_ind(psd_area_instance_dict[model_key][band][pop_name],
                                                  psd_area_instance_dict[control_model_key][band][pop_name])
                across_stat_list.append(this_stat)
                across_p_val_list.append(this_p_val)
                across_label_list.append(this_label)
    rejected, corrected_p_val_list = fdrcorrection(within_p_val_list + across_p_val_list)
    within_p_val_list = corrected_p_val_list[:within_count]
    across_p_val_list = corrected_p_val_list[within_count:]
    print('\n%s psd_area; %i comparisons:' % (band, len(corrected_p_val_list)))
    for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
        print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
    for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
        print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
band = 'Theta'
for model_key in ordered_model_key_list:
    if model_key not in nested_gamma_psd_area_instance_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s < %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = ttest_rel(nested_gamma_psd_area_instance_dict[model_key][band][pop_name], \
                                              nested_gamma_psd_area_instance_dict[model_key][band][
                                                  control_pop_name], \
                                              alternative='greater')
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list)
print('\ntheta_nested_gamma_psd_area; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, corrected_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in spatial_mod_depth_instances_dict:
        continue
    for pop_name in ['E', 'I']:
        within_count += 1
        this_label = '%s; %s; actual < predicted' % (model_key, pop_name)
        this_stat, this_p_val = ttest_rel(spatial_mod_depth_instances_dict[model_key]['actual'][pop_name], \
                                          spatial_mod_depth_instances_dict[model_key]['predicted'][pop_name], \
                                          alternative='greater')
        within_stat_list.append(this_stat)
        within_p_val_list.append(this_p_val)
        within_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list)
print('\nspatial_mod_depth; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, corrected_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in delta_peak_locs_instances_dict:
        continue
    for pop_name in ['E', 'I']:
        within_count += 1
        this_label = '%s; %s == 0' % (model_key, pop_name)
        this_stat, this_p_val = ttest_1samp(delta_peak_locs_instances_dict[model_key][pop_name], 0.)
        within_stat_list.append(this_stat)
        within_p_val_list.append(this_p_val)
        within_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list)
print('\ndelta_peak_locs; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, corrected_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in run_decoded_pos_error_instance_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s < %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = ttest_rel(run_decoded_pos_error_instance_dict[model_key][pop_name], \
                                              run_decoded_pos_error_instance_dict[model_key][control_pop_name], \
                                              alternative='greater')
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in run_decoded_pos_error_instance_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = ttest_ind(run_decoded_pos_error_instance_dict[model_key][pop_name], \
                                              run_decoded_pos_error_instance_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\nrun_decoded_position_error; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in theta_sequence_score_instances_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s < %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = ttest_rel(theta_sequence_score_instances_dict[model_key][pop_name], \
                                              theta_sequence_score_instances_dict[model_key][control_pop_name], \
                                              alternative='greater')
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in theta_sequence_score_instances_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = ttest_ind(theta_sequence_score_instances_dict[model_key][pop_name], \
                                              theta_sequence_score_instances_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\ntheta_sequence_score; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_pos_instance_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s == %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = kstest(replay_decoded_pos_instance_dict[model_key][pop_name], \
                                           replay_decoded_pos_instance_dict[model_key][control_pop_name])
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_pos_instance_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = kstest(replay_decoded_pos_instance_dict[model_key][pop_name], \
                                           replay_decoded_pos_instance_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\nreplay_decoded_pos; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_path_len_instance_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s == %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = kstest(replay_decoded_path_len_instance_dict[model_key][pop_name], \
                                           replay_decoded_path_len_instance_dict[model_key][control_pop_name])
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_path_len_instance_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = kstest(replay_decoded_path_len_instance_dict[model_key][pop_name], \
                                           replay_decoded_path_len_instance_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\nreplay_decoded_path_length; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_velocity_instance_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s == %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = kstest(replay_decoded_velocity_instance_dict[model_key][pop_name], \
                                           replay_decoded_velocity_instance_dict[model_key][control_pop_name])
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in replay_decoded_velocity_instance_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = kstest(replay_decoded_velocity_instance_dict[model_key][pop_name], \
                                           replay_decoded_velocity_instance_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\nreplay_decoded_velocity; %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  ks-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

within_count = 0
within_label_list, within_stat_list, within_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in offline_sequence_fraction_instances_dict:
        continue
    for control_pop_name in ['FF']:
        for pop_name in ['E', 'I']:
            within_count += 1
            this_label = '%s; %s < %s' % (model_key, pop_name, control_pop_name)
            this_stat, this_p_val = ttest_rel(offline_sequence_fraction_instances_dict[model_key][pop_name], \
                                              offline_sequence_fraction_instances_dict[model_key][control_pop_name], \
                                              alternative='greater')
            within_stat_list.append(this_stat)
            within_p_val_list.append(this_p_val)
            within_label_list.append(this_label)
across_label_list, across_stat_list, across_p_val_list = [], [], []
for model_key in ordered_model_key_list:
    if model_key not in offline_sequence_fraction_instances_dict:
        continue
    if model_key != control_model_key:
        for pop_name in ['E', 'I']:
            this_label = '%s; %s; %s == %s' % (model_key, pop_name, model_key, control_model_key)
            this_stat, this_p_val = ttest_ind(offline_sequence_fraction_instances_dict[model_key][pop_name], \
                                              offline_sequence_fraction_instances_dict[control_model_key][pop_name])
            across_stat_list.append(this_stat)
            across_p_val_list.append(this_p_val)
            across_label_list.append(this_label)
rejected, corrected_p_val_list = fdrcorrection(within_p_val_list+across_p_val_list)
within_p_val_list = corrected_p_val_list[:within_count]
across_p_val_list = corrected_p_val_list[within_count:]
print('\nOffline sequences (fraction of events); %i comparisons:' % (len(corrected_p_val_list)))
for this_label, this_stat, this_pval in zip(within_label_list, within_stat_list, within_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))
for this_label, this_stat, this_pval in zip(across_label_list, across_stat_list, across_p_val_list):
    print('%s;  t-stat: %.5f; corrected p: %.5f' % (this_label, this_stat, this_pval))

"""

Theta psd_area; 30 comparisons:
J; E < FF;  t-stat: 28.41354; corrected p: 0.00001
J; I < FF;  t-stat: 31.08021; corrected p: 0.00000
K; E < FF;  t-stat: 19.19826; corrected p: 0.00003
K; I < FF;  t-stat: 38.42270; corrected p: 0.00000
O_best; E < FF;  t-stat: 13.98489; corrected p: 0.00009
O_best; I < FF;  t-stat: 19.18704; corrected p: 0.00003
N; E < FF;  t-stat: 61.06612; corrected p: 0.00000
N; I < FF;  t-stat: 101.88101; corrected p: 0.00000
M_best; E < FF;  t-stat: 30.78520; corrected p: 0.00000
M_best; I < FF;  t-stat: -5.62126; corrected p: 0.99754
P_best; E < FF;  t-stat: 57.50512; corrected p: 0.00000
P_best; I < FF;  t-stat: 110.27040; corrected p: 0.00000
L; E < FF;  t-stat: 11.91070; corrected p: 0.00016
L; I < FF;  t-stat: 15.14276; corrected p: 0.00007
J_reward; E < FF;  t-stat: 45.71026; corrected p: 0.00000
J_reward; I < FF;  t-stat: 68.22504; corrected p: 0.00000
K; E; K == J;  t-stat: -13.41495; corrected p: 0.00000
K; I; K == J;  t-stat: -23.71028; corrected p: 0.00000
O_best; E; O_best == J;  t-stat: -12.09016; corrected p: 0.00000
O_best; I; O_best == J;  t-stat: -28.32222; corrected p: 0.00000
N; E; N == J;  t-stat: 23.69993; corrected p: 0.00000
N; I; N == J;  t-stat: -12.12766; corrected p: 0.00000
M_best; E; M_best == J;  t-stat: -25.05674; corrected p: 0.00000
M_best; I; M_best == J;  t-stat: -31.07934; corrected p: 0.00000
P_best; E; P_best == J;  t-stat: 52.34635; corrected p: 0.00000
P_best; I; P_best == J;  t-stat: 12.77603; corrected p: 0.00000
L; E; L == J;  t-stat: -2.61040; corrected p: 0.03333
L; I; L == J;  t-stat: -12.78516; corrected p: 0.00000
J_reward; E; J_reward == J;  t-stat: 3.99824; corrected p: 0.00440
J_reward; I; J_reward == J;  t-stat: 1.54999; corrected p: 0.16525

Gamma psd_area; 30 comparisons:
J; E < FF;  t-stat: 51.06282; corrected p: 0.00000
J; I < FF;  t-stat: 104.15859; corrected p: 0.00000
K; E < FF;  t-stat: 73.04765; corrected p: 0.00000
K; I < FF;  t-stat: 153.81355; corrected p: 0.00000
O_best; E < FF;  t-stat: 20.56833; corrected p: 0.00002
O_best; I < FF;  t-stat: 136.31681; corrected p: 0.00000
N; E < FF;  t-stat: 61.82383; corrected p: 0.00000
N; I < FF;  t-stat: 235.95416; corrected p: 0.00000
M_best; E < FF;  t-stat: 33.76045; corrected p: 0.00000
M_best; I < FF;  t-stat: 17.87343; corrected p: 0.00003
P_best; E < FF;  t-stat: 41.74411; corrected p: 0.00000
P_best; I < FF;  t-stat: 188.90563; corrected p: 0.00000
L; E < FF;  t-stat: -3.41792; corrected p: 0.98658
L; I < FF;  t-stat: 19.87594; corrected p: 0.00002
J_reward; E < FF;  t-stat: 18.63288; corrected p: 0.00003
J_reward; I < FF;  t-stat: 45.48903; corrected p: 0.00000
K; E; K == J;  t-stat: -16.30242; corrected p: 0.00000
K; I; K == J;  t-stat: 120.74831; corrected p: 0.00000
O_best; E; O_best == J;  t-stat: -9.82998; corrected p: 0.00001
O_best; I; O_best == J;  t-stat: -62.70131; corrected p: 0.00000
N; E; N == J;  t-stat: 30.36945; corrected p: 0.00000
N; I; N == J;  t-stat: -22.00159; corrected p: 0.00000
M_best; E; M_best == J;  t-stat: -49.41247; corrected p: 0.00000
M_best; I; M_best == J;  t-stat: -103.98555; corrected p: 0.00000
P_best; E; P_best == J;  t-stat: 39.17775; corrected p: 0.00000
P_best; I; P_best == J;  t-stat: 36.03953; corrected p: 0.00000
L; E; L == J;  t-stat: -52.99500; corrected p: 0.00000
L; I; L == J;  t-stat: -69.95538; corrected p: 0.00000
J_reward; E; J_reward == J;  t-stat: 5.80553; corrected p: 0.00043
J_reward; I; J_reward == J;  t-stat: 3.02688; corrected p: 0.01695

Ripple psd_area; 30 comparisons:
J; E < FF;  t-stat: -21.45982; corrected p: 1.00000
J; I < FF;  t-stat: 67.57158; corrected p: 0.00000
K; E < FF;  t-stat: -1.37451; corrected p: 0.97708
K; I < FF;  t-stat: 85.91006; corrected p: 0.00000
O_best; E < FF;  t-stat: 7.96400; corrected p: 0.00092
O_best; I < FF;  t-stat: 19.43027; corrected p: 0.00004
N; E < FF;  t-stat: 7.70488; corrected p: 0.00100
N; I < FF;  t-stat: 35.25801; corrected p: 0.00001
M_best; E < FF;  t-stat: -56.58242; corrected p: 1.00000
M_best; I < FF;  t-stat: -212.96441; corrected p: 1.00000
P_best; E < FF;  t-stat: 160.04623; corrected p: 0.00000
P_best; I < FF;  t-stat: 75.89306; corrected p: 0.00000
L; E < FF;  t-stat: 7.35491; corrected p: 0.00114
L; I < FF;  t-stat: 21.86833; corrected p: 0.00003
J_reward; E < FF;  t-stat: 8.74698; corrected p: 0.00067
J_reward; I < FF;  t-stat: 16.93985; corrected p: 0.00006
K; E; K == J;  t-stat: -0.60352; corrected p: 0.64948
K; I; K == J;  t-stat: 73.98532; corrected p: 0.00000
O_best; E; O_best == J;  t-stat: 8.23369; corrected p: 0.00006
O_best; I; O_best == J;  t-stat: 5.82657; corrected p: 0.00059
N; E; N == J;  t-stat: 10.91325; corrected p: 0.00001
N; I; N == J;  t-stat: 2.86756; corrected p: 0.02509
M_best; E; M_best == J;  t-stat: -12.00189; corrected p: 0.00001
M_best; I; M_best == J;  t-stat: -69.49431; corrected p: 0.00000
P_best; E; P_best == J;  t-stat: 160.58677; corrected p: 0.00000
P_best; I; P_best == J;  t-stat: 64.61204; corrected p: 0.00000
L; E; L == J;  t-stat: 14.73887; corrected p: 0.00000
L; I; L == J;  t-stat: 6.57871; corrected p: 0.00027
J_reward; E; J_reward == J;  t-stat: 9.40200; corrected p: 0.00003
J_reward; I; J_reward == J;  t-stat: 11.36043; corrected p: 0.00001

theta_nested_gamma_psd_area; 16 comparisons:
J; E < FF;  t-stat: 23.62390; corrected p: 0.00002
J; I < FF;  t-stat: 23.39898; corrected p: 0.00002
K; E < FF;  t-stat: 10.47157; corrected p: 0.00027
K; I < FF;  t-stat: 23.86003; corrected p: 0.00002
O_best; E < FF;  t-stat: 11.60964; corrected p: 0.00019
O_best; I < FF;  t-stat: 53.14665; corrected p: 0.00000
N; E < FF;  t-stat: 41.34734; corrected p: 0.00000
N; I < FF;  t-stat: 166.51610; corrected p: 0.00000
M_best; E < FF;  t-stat: 7.23216; corrected p: 0.00097
M_best; I < FF;  t-stat: 19.99111; corrected p: 0.00003
P_best; E < FF;  t-stat: 20.36267; corrected p: 0.00003
P_best; I < FF;  t-stat: 31.72323; corrected p: 0.00001
L; E < FF;  t-stat: 9.11079; corrected p: 0.00043
L; I < FF;  t-stat: 13.79812; corrected p: 0.00011
J_reward; E < FF;  t-stat: 22.44379; corrected p: 0.00002
J_reward; I < FF;  t-stat: 53.76653; corrected p: 0.00000

spatial_mod_depth; 16 comparisons:
J; E; actual < predicted;  t-stat: 476.78950; corrected p: 0.00000
J; I; actual < predicted;  t-stat: 65.77625; corrected p: 0.00000
K; E; actual < predicted;  t-stat: 291.44528; corrected p: 0.00000
K; I; actual < predicted;  t-stat: 47.76743; corrected p: 0.00000
O_best; E; actual < predicted;  t-stat: 119.39097; corrected p: 0.00000
O_best; I; actual < predicted;  t-stat: 49.62370; corrected p: 0.00000
N; E; actual < predicted;  t-stat: 303.27876; corrected p: 0.00000
N; I; actual < predicted;  t-stat: 86.68804; corrected p: 0.00000
M_best; E; actual < predicted;  t-stat: 1136.84655; corrected p: 0.00000
M_best; I; actual < predicted;  t-stat: 133.77878; corrected p: 0.00000
P_best; E; actual < predicted;  t-stat: 145.14421; corrected p: 0.00000
P_best; I; actual < predicted;  t-stat: 79.75489; corrected p: 0.00000
L; E; actual < predicted;  t-stat: 117.86428; corrected p: 0.00000
L; I; actual < predicted;  t-stat: 97.78824; corrected p: 0.00000
J_reward; E; actual < predicted;  t-stat: 274.79937; corrected p: 0.00000
J_reward; I; actual < predicted;  t-stat: 42.24290; corrected p: 0.00000

delta_peak_locs; 16 comparisons:
J; E == 0;  t-stat: 138.37732; corrected p: 0.00000
J; I == 0;  t-stat: 19.51767; corrected p: 0.00005
K; E == 0;  t-stat: 268.83081; corrected p: 0.00000
K; I == 0;  t-stat: 13.51897; corrected p: 0.00017
O_best; E == 0;  t-stat: 65.23775; corrected p: 0.00000
O_best; I == 0;  t-stat: 15.84090; corrected p: 0.00010
N; E == 0;  t-stat: 56.17979; corrected p: 0.00000
N; I == 0;  t-stat: 29.44912; corrected p: 0.00001
M_best; E == 0;  t-stat: 128.63164; corrected p: 0.00000
M_best; I == 0;  t-stat: 29.09395; corrected p: 0.00001
P_best; E == 0;  t-stat: 26.43846; corrected p: 0.00002
P_best; I == 0;  t-stat: 23.20399; corrected p: 0.00003
L; E == 0;  t-stat: 45.59159; corrected p: 0.00000
L; I == 0;  t-stat: 20.49575; corrected p: 0.00004
J_reward; E == 0;  t-stat: 54.36568; corrected p: 0.00000
J_reward; I == 0;  t-stat: 22.70318; corrected p: 0.00003

run_decoded_position_error; 10 comparisons:
J; E < FF;  t-stat: 49.54661; corrected p: 0.00000
J; I < FF;  t-stat: 36.24436; corrected p: 0.00000
K; E < FF;  t-stat: 21.11278; corrected p: 0.00001
K; I < FF;  t-stat: 24.06379; corrected p: 0.00001
O_best; E < FF;  t-stat: 25.90706; corrected p: 0.00001
O_best; I < FF;  t-stat: 28.59301; corrected p: 0.00001
K; E; K == J;  t-stat: -33.70574; corrected p: 0.00000
K; I; K == J;  t-stat: -9.40496; corrected p: 0.00001
O_best; E; O_best == J;  t-stat: -21.85687; corrected p: 0.00000
O_best; I; O_best == J;  t-stat: -9.65608; corrected p: 0.00001

theta_sequence_score; 10 comparisons:
J; E < FF;  t-stat: 16.86711; corrected p: 0.00005
J; I < FF;  t-stat: 18.12250; corrected p: 0.00005
K; E < FF;  t-stat: 24.24924; corrected p: 0.00002
K; I < FF;  t-stat: 0.72395; corrected p: 0.28287
O_best; E < FF;  t-stat: 13.63185; corrected p: 0.00010
O_best; I < FF;  t-stat: -4.90938; corrected p: 0.99600
K; E; K == J;  t-stat: -14.57945; corrected p: 0.00000
K; I; K == J;  t-stat: -13.86876; corrected p: 0.00000
O_best; E; O_best == J;  t-stat: -12.69798; corrected p: 0.00000
O_best; I; O_best == J;  t-stat: -14.45422; corrected p: 0.00000

replay_decoded_pos; 30 comparisons:
J; E == FF;  ks-stat: 0.02200; corrected p: 0.99974
J; I == FF;  ks-stat: 0.03000; corrected p: 0.99974
K; E == FF;  ks-stat: 0.05000; corrected p: 0.99974
K; I == FF;  ks-stat: 0.03800; corrected p: 0.99974
O_best; E == FF;  ks-stat: 0.04600; corrected p: 0.99974
O_best; I == FF;  ks-stat: 0.03000; corrected p: 0.99974
N; E == FF;  ks-stat: 0.02400; corrected p: 0.99974
N; I == FF;  ks-stat: 0.04400; corrected p: 0.99974
M_best; E == FF;  ks-stat: 0.02400; corrected p: 0.99974
M_best; I == FF;  ks-stat: 0.02400; corrected p: 0.99974
P_best; E == FF;  ks-stat: 0.15000; corrected p: 0.00019
P_best; I == FF;  ks-stat: 0.04000; corrected p: 0.99974
L; E == FF;  ks-stat: 0.05000; corrected p: 0.99974
L; I == FF;  ks-stat: 0.06000; corrected p: 0.89825
J_reward; E == FF;  ks-stat: 0.20400; corrected p: 0.00000
J_reward; I == FF;  ks-stat: 0.08200; corrected p: 0.41580
K; E; K == J;  ks-stat: 0.06400; corrected p: 0.85869
K; I; K == J;  ks-stat: 0.06200; corrected p: 0.87577
O_best; E; O_best == J;  ks-stat: 0.04000; corrected p: 0.99974
O_best; I; O_best == J;  ks-stat: 0.03800; corrected p: 0.99974
N; E; N == J;  ks-stat: 0.02600; corrected p: 0.99974
N; I; N == J;  ks-stat: 0.04600; corrected p: 0.99974
M_best; E; M_best == J;  ks-stat: 0.03400; corrected p: 0.99974
M_best; I; M_best == J;  ks-stat: 0.03200; corrected p: 0.99974
P_best; E; P_best == J;  ks-stat: 0.15600; corrected p: 0.00010
P_best; I; P_best == J;  ks-stat: 0.04600; corrected p: 0.99974
L; E; L == J;  ks-stat: 0.06800; corrected p: 0.84875
L; I; L == J;  ks-stat: 0.06400; corrected p: 0.85869
J_reward; E; J_reward == J;  ks-stat: 0.18600; corrected p: 0.00000
J_reward; I; J_reward == J;  ks-stat: 0.07000; corrected p: 0.84875

replay_decoded_path_length; 30 comparisons:
J; E == FF;  ks-stat: 0.81600; corrected p: 0.00000
J; I == FF;  ks-stat: 0.31600; corrected p: 0.00000
K; E == FF;  ks-stat: 0.34400; corrected p: 0.00000
K; I == FF;  ks-stat: 0.03000; corrected p: 0.99997
O_best; E == FF;  ks-stat: 0.29000; corrected p: 0.00000
O_best; I == FF;  ks-stat: 0.07800; corrected p: 0.11015
N; E == FF;  ks-stat: 0.83200; corrected p: 0.00000
N; I == FF;  ks-stat: 0.34600; corrected p: 0.00000
M_best; E == FF;  ks-stat: 0.43000; corrected p: 0.00000
M_best; I == FF;  ks-stat: 0.88000; corrected p: 0.00000
P_best; E == FF;  ks-stat: 0.36000; corrected p: 0.00000
P_best; I == FF;  ks-stat: 0.13400; corrected p: 0.00030
L; E == FF;  ks-stat: 0.56000; corrected p: 0.00000
L; I == FF;  ks-stat: 0.30800; corrected p: 0.00000
J_reward; E == FF;  ks-stat: 0.81800; corrected p: 0.00000
J_reward; I == FF;  ks-stat: 0.50400; corrected p: 0.00000
K; E; K == J;  ks-stat: 0.65600; corrected p: 0.00000
K; I; K == J;  ks-stat: 0.31000; corrected p: 0.00000
O_best; E; O_best == J;  ks-stat: 0.67000; corrected p: 0.00000
O_best; I; O_best == J;  ks-stat: 0.37200; corrected p: 0.00000
N; E; N == J;  ks-stat: 0.18000; corrected p: 0.00000
N; I; N == J;  ks-stat: 0.06200; corrected p: 0.31278
M_best; E; M_best == J;  ks-stat: 0.60200; corrected p: 0.00000
M_best; I; M_best == J;  ks-stat: 0.74800; corrected p: 0.00000
P_best; E; P_best == J;  ks-stat: 0.51600; corrected p: 0.00000
P_best; I; P_best == J;  ks-stat: 0.42200; corrected p: 0.00000
L; E; L == J;  ks-stat: 0.38000; corrected p: 0.00000
L; I; L == J;  ks-stat: 0.02000; corrected p: 0.99997
J_reward; E; J_reward == J;  ks-stat: 0.06600; corrected p: 0.25150
J_reward; I; J_reward == J;  ks-stat: 0.20000; corrected p: 0.00000

replay_decoded_velocity; 30 comparisons:
J; E == FF;  ks-stat: 0.12600; corrected p: 0.00235
J; I == FF;  ks-stat: 0.05000; corrected p: 0.73515
K; E == FF;  ks-stat: 0.02600; corrected p: 0.99877
K; I == FF;  ks-stat: 0.02400; corrected p: 0.99877
O_best; E == FF;  ks-stat: 0.03600; corrected p: 0.96717
O_best; I == FF;  ks-stat: 0.04800; corrected p: 0.73515
N; E == FF;  ks-stat: 0.38600; corrected p: 0.00000
N; I == FF;  ks-stat: 0.14400; corrected p: 0.00026
M_best; E == FF;  ks-stat: 0.05000; corrected p: 0.73515
M_best; I == FF;  ks-stat: 0.17000; corrected p: 0.00001
P_best; E == FF;  ks-stat: 0.15000; corrected p: 0.00015
P_best; I == FF;  ks-stat: 0.06800; corrected p: 0.31270
L; E == FF;  ks-stat: 0.08000; corrected p: 0.15282
L; I == FF;  ks-stat: 0.04800; corrected p: 0.73515
J_reward; E == FF;  ks-stat: 0.10200; corrected p: 0.02742
J_reward; I == FF;  ks-stat: 0.04800; corrected p: 0.73515
K; E; K == J;  ks-stat: 0.10800; corrected p: 0.01591
K; I; K == J;  ks-stat: 0.04600; corrected p: 0.76837
O_best; E; O_best == J;  ks-stat: 0.12400; corrected p: 0.00272
O_best; I; O_best == J;  ks-stat: 0.07200; corrected p: 0.24955
N; E; N == J;  ks-stat: 0.33600; corrected p: 0.00000
N; I; N == J;  ks-stat: 0.14400; corrected p: 0.00026
M_best; E; M_best == J;  ks-stat: 0.08600; corrected p: 0.09901
M_best; I; M_best == J;  ks-stat: 0.16000; corrected p: 0.00004
P_best; E; P_best == J;  ks-stat: 0.13400; corrected p: 0.00093
P_best; I; P_best == J;  ks-stat: 0.09200; corrected p: 0.06213
L; E; L == J;  ks-stat: 0.10000; corrected p: 0.03099
L; I; L == J;  ks-stat: 0.04000; corrected p: 0.91017
J_reward; E; J_reward == J;  ks-stat: 0.07800; corrected p: 0.16847
J_reward; I; J_reward == J;  ks-stat: 0.05600; corrected p: 0.62023

Offline sequences (fraction of events); 30 comparisons:
J; E < FF;  t-stat: 113.23501; corrected p: 0.00000
J; I < FF;  t-stat: 8.36841; corrected p: 0.00084
K; E < FF;  t-stat: 18.37373; corrected p: 0.00005
K; I < FF;  t-stat: -0.97645; corrected p: 0.86564
O_best; E < FF;  t-stat: 1.99818; corrected p: 0.06713
O_best; I < FF;  t-stat: -9.62684; corrected p: 0.99967
N; E < FF;  t-stat: 63.30095; corrected p: 0.00000
N; I < FF;  t-stat: 21.98069; corrected p: 0.00003
M_best; E < FF;  t-stat: 19.25167; corrected p: 0.00004
M_best; I < FF;  t-stat: 23.87809; corrected p: 0.00002
P_best; E < FF;  t-stat: 1.30800; corrected p: 0.14499
P_best; I < FF;  t-stat: -6.56442; corrected p: 0.99967
L; E < FF;  t-stat: 10.43259; corrected p: 0.00038
L; I < FF;  t-stat: 7.44369; corrected p: 0.00119
J_reward; E < FF;  t-stat: 48.81826; corrected p: 0.00000
J_reward; I < FF;  t-stat: 11.65033; corrected p: 0.00027
K; E; K == J;  t-stat: -48.56098; corrected p: 0.00000
K; I; K == J;  t-stat: -10.72239; corrected p: 0.00001
O_best; E; O_best == J;  t-stat: -17.71135; corrected p: 0.00000
O_best; I; O_best == J;  t-stat: -13.66911; corrected p: 0.00000
N; E; N == J;  t-stat: 14.61835; corrected p: 0.00000
N; I; N == J;  t-stat: 5.23921; corrected p: 0.00112
M_best; E; M_best == J;  t-stat: -28.80089; corrected p: 0.00000
M_best; I; M_best == J;  t-stat: 11.82184; corrected p: 0.00001
P_best; E; P_best == J;  t-stat: -6.65926; corrected p: 0.00027
P_best; I; P_best == J;  t-stat: -14.94083; corrected p: 0.00000
L; E; L == J;  t-stat: -15.60747; corrected p: 0.00000
L; I; L == J;  t-stat: -2.61527; corrected p: 0.03860
J_reward; E; J_reward == J;  t-stat: 2.37382; corrected p: 0.05397
J_reward; I; J_reward == J;  t-stat: 3.73127; corrected p: 0.00754

"""