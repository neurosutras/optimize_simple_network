from simple_network_analysis_utils import *
from matplotlib.lines import Line2D

mpl.rcParams['font.size'] = 14.

summary_data_file_path = 'data/20211025_simple_network_instances_summary_data_for_stats.hdf5'

run_band_filter_range = {'Theta': (4., 10.), 'Gamma': (30., 100.)}

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
                          'O': ['O_18883', 'O_21309', 'O_27288', 'O_29027', 'O_best'],
                          'M': ['M_best', 'M_28922', 'M_26434', 'M_29613', 'M_28842'],
                          'P': ['P_best', 'P_24612', 'P_22746', 'P_18307', 'P_15131'],
                          'L': ['L', 'L_7820', 'L_13211', 'L_22102', 'L_27537']}

marder_group_labels_dict = {'J': 'Structured\nE <-E weights',
                            'K': 'Random\nE <-E weights',
                            'O': 'Shuffled\nE <-E weights',
                            'M': 'No rhythmicity\nconstraints',
                            'P': 'No sparsity or\nselectivity constraints',
                            'L': 'No spike rate\nadaptation'}

spatial_mod_depth_marder_group_dict = defaultdict(lambda: defaultdict(list))
fraction_active_run_marder_group_dict = defaultdict(lambda: defaultdict(list))
rhythmicity_marder_group_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
        for pop_name in offline_sequence_fraction_instances_dict[model_key]:
            offline_sequence_fraction_marder_group_dict[group_key][pop_name].append(
                offline_sequence_fraction_instances_dict[model_key][pop_name])


np.random.seed(0)

ordered_group_keys = ['J', 'K', 'O', 'M', 'P', 'L']
control_group_key = 'J'

num_comparisons = len(ordered_group_keys) - 1

xticklabels = [marder_group_labels_dict[group_key] for group_key in ordered_group_keys]

fig, axes = plt.subplots(3, 2, figsize=(10., 11.))
flat_axes = axes.flatten()
axis = 0
data = [spatial_mod_depth_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Spatial selectivity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        result = stats.ttest_ind(spatial_mod_depth_marder_group_dict[group_key]['E'],
                                    spatial_mod_depth_marder_group_dict[control_group_key]['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%.5f' % (min(1., result.pvalue * num_comparisons)))
print('\n')
for vals, group_key in zip(data, ordered_group_keys):
    print('%s; mean: %.5f, sd: %.5f' % (group_key, np.mean(vals), np.std(vals)))
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
        result = stats.ttest_ind(fraction_active_run_marder_group_dict[group_key]['E'],
                                    fraction_active_run_marder_group_dict[control_group_key]['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%.5f' % (min(1., result.pvalue * num_comparisons)))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1
print('\n')
for vals, group_key in zip(data, ordered_group_keys):
    print('%s; mean: %.5f, sd: %.5f' % (group_key, np.mean(vals), np.std(vals)))

data = [rhythmicity_marder_group_dict[group_key]['Gamma']['E'] for group_key in ordered_group_keys]
ylabel = 'Gamma rhythmicity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        result = stats.ttest_ind(rhythmicity_marder_group_dict[group_key]['Gamma']['E'],
                                    rhythmicity_marder_group_dict[control_group_key]['Gamma']['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%.5f' % (min(1., result.pvalue * num_comparisons)))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(rhythmicity_marder_group_dict[control_group_key]['Gamma']['FF']),
                     '--', c='grey')
flat_axes[axis].set_yscale('log')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylabel(ylabel)
axis += 1
print('\n')
for vals, group_key in zip(data, ordered_group_keys):
    print('%s; mean: %.5f, sd: %.5f' % (group_key, np.mean(vals), np.std(vals)))

data = [rhythmicity_marder_group_dict[group_key]['Theta']['E'] for group_key in ordered_group_keys]
ylabel = 'Theta rhythmicity'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        result = stats.ttest_ind(rhythmicity_marder_group_dict[group_key]['Theta']['E'],
                                    rhythmicity_marder_group_dict[control_group_key]['Theta']['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%.5f' % (min(1., result.pvalue * num_comparisons)))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(rhythmicity_marder_group_dict[control_group_key]['Theta']['FF']),
                     '--', c='grey')
flat_axes[axis].set_yscale('log')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylabel(ylabel)
axis += 1
print('\n')
for vals, group_key in zip(data, ordered_group_keys):
    print('%s; mean: %.5f, sd: %.5f' % (group_key, np.mean(vals), np.std(vals)))

data = [offline_sequence_fraction_marder_group_dict[group_key]['E'] for group_key in ordered_group_keys]
ylabel = 'Offline sequences\n(fraction of events)'
print('\nANOVA:', ylabel, stats.f_oneway(*data))
for group_key in ordered_group_keys:
    if group_key != control_group_key:
        result = stats.ttest_ind(offline_sequence_fraction_marder_group_dict[group_key]['E'],
                                    offline_sequence_fraction_marder_group_dict[control_group_key]['E'])
        print('unpaired t-test:', ylabel, '%s vs %s' % (group_key, control_group_key), result,
              'corrected pvalue=%.5f' % (min(1., result.pvalue * num_comparisons)))
scattered_boxplot(flat_axes[axis], data, showfliers='unif')
flat_axes[axis].plot(flat_axes[axis].get_xlim(),
                     np.ones(2) * np.mean(offline_sequence_fraction_marder_group_dict[control_group_key]['FF']),
                     '--', c='grey')
flat_axes[axis].set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
flat_axes[axis].set_ylim(0., 1.1*flat_axes[axis].get_ylim()[1])
flat_axes[axis].set_ylabel(ylabel)
axis += 1
print('\n')
for vals, group_key in zip(data, ordered_group_keys):
    print('%s; mean: %.5f, sd: %.5f' % (group_key, np.mean(vals), np.std(vals)))

fig.tight_layout(h_pad=1.)
fig.show()

clean_axes(flat_axes)

plt.show()

"""

ANOVA: Spatial selectivity F_onewayResult(statistic=66.25479731100481, pvalue=2.9739170969548613e-13)
unpaired t-test: Spatial selectivity K vs J Ttest_indResult(statistic=0.24552797278222963, pvalue=0.8122291505151493) corrected pvalue=4.061145752575746
unpaired t-test: Spatial selectivity O vs J Ttest_indResult(statistic=-5.713346753037892, pvalue=0.0004474465505885665) corrected pvalue=0.0022372327529428327
unpaired t-test: Spatial selectivity M vs J Ttest_indResult(statistic=0.7388472234900785, pvalue=0.48111221334808285) corrected pvalue=2.405561066740414
unpaired t-test: Spatial selectivity P vs J Ttest_indResult(statistic=-87.53901690832106, pvalue=3.2357379254521646e-13) corrected pvalue=1.6178689627260823e-12
unpaired t-test: Spatial selectivity L vs J Ttest_indResult(statistic=-0.24479724512542633, pvalue=0.8127755753991768) corrected pvalue=4.063877876995884

ANOVA: Active fraction F_onewayResult(statistic=775.6183987028032, pvalue=1.0537805966610357e-25)
unpaired t-test: Active fraction K vs J Ttest_indResult(statistic=1.1116099374699027, pvalue=0.2985934570048069) corrected pvalue=1.4929672850240343
unpaired t-test: Active fraction O vs J Ttest_indResult(statistic=2.773727744565321, pvalue=0.024157273701560657) corrected pvalue=0.12078636850780328
unpaired t-test: Active fraction M vs J Ttest_indResult(statistic=0.04767600237968934, pvalue=0.9631431803673343) corrected pvalue=4.815715901836671
unpaired t-test: Active fraction P vs J Ttest_indResult(statistic=66.05485814816694, pvalue=3.0698410449810925e-12) corrected pvalue=1.5349205224905462e-11
unpaired t-test: Active fraction L vs J Ttest_indResult(statistic=-2.4639590563904363, pvalue=0.03907677964383308) corrected pvalue=0.1953838982191654

ANOVA: Gamma rhythmicity F_onewayResult(statistic=45.30661635855427, pvalue=1.885144747783006e-11)
unpaired t-test: Gamma rhythmicity K vs J Ttest_indResult(statistic=-3.5881805188623805, pvalue=0.007103252212579735) corrected pvalue=0.03551626106289867
unpaired t-test: Gamma rhythmicity O vs J Ttest_indResult(statistic=-2.577796694300953, pvalue=0.0327282672396406) corrected pvalue=0.16364133619820298
unpaired t-test: Gamma rhythmicity M vs J Ttest_indResult(statistic=-8.514649787845457, pvalue=2.7802851231962653e-05) corrected pvalue=0.00013901425615981327
unpaired t-test: Gamma rhythmicity P vs J Ttest_indResult(statistic=6.531094006702287, pvalue=0.00018204999834642638) corrected pvalue=0.0009102499917321319
unpaired t-test: Gamma rhythmicity L vs J Ttest_indResult(statistic=-8.67896275552181, pvalue=2.418408737982482e-05) corrected pvalue=0.0001209204368991241

ANOVA: Theta rhythmicity F_onewayResult(statistic=21.03983226573771, pvalue=4.617599202898925e-08)
unpaired t-test: Theta rhythmicity K vs J Ttest_indResult(statistic=-5.544471367900092, pvalue=0.0005444942238575037) corrected pvalue=0.0027224711192875184
unpaired t-test: Theta rhythmicity O vs J Ttest_indResult(statistic=-4.01120578256563, pvalue=0.003888515778452034) corrected pvalue=0.01944257889226017
unpaired t-test: Theta rhythmicity M vs J Ttest_indResult(statistic=-7.663559927505312, pvalue=5.941532138004858e-05) corrected pvalue=0.0002970766069002429
unpaired t-test: Theta rhythmicity P vs J Ttest_indResult(statistic=4.512760497075627, pvalue=0.0019686244270386718) corrected pvalue=0.00984312213519336
unpaired t-test: Theta rhythmicity L vs J Ttest_indResult(statistic=1.359251361065622, pvalue=0.21114163231904898) corrected pvalue=1.0557081615952448

ANOVA: Offline sequences
(fraction of events) F_onewayResult(statistic=18.758429636739816, pvalue=1.3581827360578167e-07)
unpaired t-test: Offline sequences
(fraction of events) K vs J Ttest_indResult(statistic=-16.356537065251764, pvalue=1.9661189861443126e-07) corrected pvalue=9.830594930721563e-07
unpaired t-test: Offline sequences
(fraction of events) O vs J Ttest_indResult(statistic=-7.241487666450048, pvalue=8.87857027221006e-05) corrected pvalue=0.000443928513610503
unpaired t-test: Offline sequences
(fraction of events) M vs J Ttest_indResult(statistic=-12.884923160809848, pvalue=1.244386725663577e-06) corrected pvalue=6.221933628317885e-06
unpaired t-test: Offline sequences
(fraction of events) P vs J Ttest_indResult(statistic=-6.267963072679115, pvalue=0.00024097817884415206) corrected pvalue=0.0012048908942207603
unpaired t-test: Offline sequences
(fraction of events) L vs J Ttest_indResult(statistic=-4.719972111289305, pvalue=0.001502218931162961) corrected pvalue=0.007511094655814805


"""