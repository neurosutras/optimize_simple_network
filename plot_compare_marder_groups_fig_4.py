from simple_network_analysis_utils import *
from matplotlib.lines import Line2D

mpl.rcParams['font.size'] = 14.

summary_data_file_path = 'data/20211025_simple_network_instances_summary_data_for_stats.hdf5'

run_band_filter_range = {'Theta': (4., 10.), 'Gamma': (30., 100.)}

theta_sequence_score_instances_dict = defaultdict(lambda: defaultdict(list))
spatial_mod_depth_instances_dict = defaultdict(dict)
fraction_active_run_instances_dict = defaultdict(dict)
psd_area_instances_dict = defaultdict(
    lambda: {'Theta': defaultdict(list), 'Gamma': defaultdict(list), 'Ripple': defaultdict(list)})
offline_sequence_fraction_instances_dict = defaultdict(dict)

with h5py.File(summary_data_file_path, 'r') as f:
    fft_f = f['shared_context']['fft_f'][:]
    for model_key in f:
        if model_key == 'shared_context':
            continue

        if 'theta_sequence_score' in f[model_key]:
            for pop_name in f[model_key]['theta_sequence_score']:
                for instance in f[model_key]['theta_sequence_score'][pop_name].values():
                    this_mean_seq_score = np.mean(instance[:])
                    theta_sequence_score_instances_dict[model_key][pop_name].append(this_mean_seq_score)

        if 'spatial_modulation_depth' in f[model_key]:
            condition = 'actual'
            for pop_name in f[model_key]['spatial_modulation_depth'][condition]:
                spatial_mod_depth_instances_dict[model_key][pop_name] = \
                    f[model_key]['spatial_modulation_depth'][condition][pop_name][:]

        if 'fraction_active_run' in f[model_key]:
            for pop_name in f[model_key]['fraction_active_run']:
                fraction_active_run_instances_dict[model_key][pop_name] = \
                    f[model_key]['fraction_active_run'][pop_name][:]

        if 'fft_power' in f[model_key]:
            for pop_name in f[model_key]['fft_power']:
                for instance in f[model_key]['fft_power'][pop_name][:]:
                    d_fft_f = fft_f[1] - fft_f[0]
                    for band, this_band_filter_range in run_band_filter_range.items():
                        indexes = np.where((fft_f >= this_band_filter_range[0]) & (fft_f <= this_band_filter_range[1]))
                        this_area = np.trapz(instance[indexes], dx=d_fft_f)
                        psd_area_instances_dict[model_key][band][pop_name].append(this_area)

        if 'replay_sequence_fraction' in f[model_key]:
            for pop_name in f[model_key]['replay_sequence_fraction']:
                offline_sequence_fraction_instances_dict[model_key][pop_name] = np.mean(
                    f[model_key]['replay_sequence_fraction'][pop_name][:])

marder_group_keys_dict = {'J': ['J', 'J_25866', 'J_26467', 'J_29427', 'J_29623'],
                          'K': ['K', 'K_12596', 'K_16954', 'K_27171', 'K_9626'],
                          'O': ['O_18883', 'O_21309', 'O_27288', 'O_29027', 'O_best']}

marder_group_labels_dict = {'J': 'Structured\nE <-E weights',
                            'K': 'Random\nE <-E weights',
                            'O': 'Shuffled\nE <-E weights'}

spatial_mod_depth_marder_group_dict = defaultdict(lambda: defaultdict(list))
fraction_active_run_marder_group_dict = defaultdict(lambda: defaultdict(list))
rhythmicity_marder_group_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
theta_sequence_score_marder_group_dict = defaultdict(lambda: defaultdict(list))
offline_sequence_fraction_marder_group_dict = defaultdict(lambda: defaultdict(list))

for group_key in marder_group_keys_dict:
    for model_key in marder_group_keys_dict[group_key]:
        for pop_name in spatial_mod_depth_instances_dict[model_key]:
            spatial_mod_depth_marder_group_dict[group_key][pop_name].append(
                np.mean(spatial_mod_depth_instances_dict[model_key][pop_name]))

for group_key in marder_group_keys_dict:
    for model_key in marder_group_keys_dict[group_key]:
        for pop_name in fraction_active_run_instances_dict[model_key]:
            fraction_active_run_marder_group_dict[group_key][pop_name].append(
                np.mean(fraction_active_run_instances_dict[model_key][pop_name]))

for group_key in marder_group_keys_dict:
    for model_key in marder_group_keys_dict[group_key]:
        for band in psd_area_instances_dict[model_key]:
            for pop_name in psd_area_instances_dict[model_key][band]:
                rhythmicity_marder_group_dict[group_key][band][pop_name].append(
                    np.mean(psd_area_instances_dict[model_key][band][pop_name]))

for group_key in marder_group_keys_dict:
    for model_key in marder_group_keys_dict[group_key]:
        for pop_name in theta_sequence_score_instances_dict[model_key]:
            theta_sequence_score_marder_group_dict[group_key][pop_name].append(
                np.mean(theta_sequence_score_instances_dict[model_key][pop_name]))

for group_key in marder_group_keys_dict:
    for model_key in marder_group_keys_dict[group_key]:
        for pop_name in offline_sequence_fraction_instances_dict[model_key]:
            offline_sequence_fraction_marder_group_dict[group_key][pop_name].append(
                offline_sequence_fraction_instances_dict[model_key][pop_name])


ordered_group_keys = ['J', 'K', 'O']
control_group_key = 'J'

num_comparisons = len(ordered_group_keys) - 1

xticklabels = [marder_group_labels_dict[group_key] for group_key in ordered_group_keys]

fig, axes = plt.subplots(2, 3, figsize=(15., 9.))
flat_axes = axes.flatten()
axis = 0
data = [spatial_mod_depth_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Spatial selectivity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key),
              stats.ttest_ind(spatial_mod_depth_marder_group_dict[group_key]['E'],
                                 spatial_mod_depth_marder_group_dict[control_group_key]['E']))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1

data = [fraction_active_run_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Active fraction'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key),
              stats.ttest_ind(fraction_active_run_marder_group_dict[group_key]['E'],
                                 fraction_active_run_marder_group_dict[control_group_key]['E']))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1

data = [rhythmicity_marder_group_dict[group_key]['Gamma']['E'] for group_key in ordered_group_keys]
ylabel = 'Gamma rhythmicity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key),
              stats.ttest_ind(rhythmicity_marder_group_dict[group_key]['Gamma']['E'],
                                 rhythmicity_marder_group_dict[control_group_key]['Gamma']['E']))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(rhythmicity_marder_group_dict[control_group_key]['Gamma']['FF']),
                     '--', c='grey')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1

data = [rhythmicity_marder_group_dict[group_key]['Theta']['E'] for group_key in ordered_group_keys]
ylabel = 'Theta rhythmicity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key),
              stats.ttest_ind(rhythmicity_marder_group_dict[group_key]['Theta']['E'],
                                 rhythmicity_marder_group_dict[control_group_key]['Theta']['E']))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(rhythmicity_marder_group_dict[control_group_key]['Theta']['FF']),
                     '--', c='grey')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1

data = [theta_sequence_score_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Theta sequence score'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        result = stats.ttest_ind(theta_sequence_score_marder_group_dict[group_key]['E'],
                                    theta_sequence_score_marder_group_dict[control_group_key]['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%s' % (result.pvalue * num_comparisons))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(theta_sequence_score_marder_group_dict[control_group_key]['FF']),
                     '--', c='grey')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)

axis += 1
data = [offline_sequence_fraction_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Offline sequences\n(fraction of events)'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key),
              stats.ttest_ind(offline_sequence_fraction_marder_group_dict[group_key]['E'],
                                 offline_sequence_fraction_marder_group_dict[control_group_key]['E']))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(offline_sequence_fraction_marder_group_dict[control_group_key]['FF']),
                     '--', c='grey')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)

fig.tight_layout(w_pad=1.9, h_pad=1.9)
fig.show()

clean_axes(flat_axes)

plt.show()


"""

ANOVA: Theta sequence score F_onewayResult(statistic=11.191924798097315, pvalue=0.0018070097075653907)
unpaired t-test: Theta sequence score K vs J Ttest_indResult(statistic=-3.126641079467246, pvalue=0.014086586716873446) corrected pvalue=0.028173173433746892
unpaired t-test: Theta sequence score O vs J Ttest_indResult(statistic=-3.563218355218198, pvalue=0.0073660879134724595) corrected pvalue=0.014732175826944919

"""