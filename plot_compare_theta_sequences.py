from simple_network_analysis_utils import *
from matplotlib.lines import Line2D

summary_data_file_path = 'data/20211025_simple_network_instances_summary_data_for_stats.hdf5'

pop_order = ['FF', 'E', 'I']
color_dict = {'FF': 'darkgrey',
              'E': 'r',
              'I': 'b'}

ordered_model_key_list = ['J', 'K', 'O_best']
instance_index_dict = {'J': 0,
                       'K': 0,
                       'O_best': 0}

model_label_dict = {'J': 'Structured\nE<-E weights',
                    'K': 'Random\nE<-E weights',
                    'O_best': 'Shuffled\nE<-E weights'}

decoded_pos_error_dict = defaultdict(lambda: defaultdict(list))
theta_sequence_score_dict = defaultdict(lambda: defaultdict(list))

with h5py.File(summary_data_file_path, 'r') as f:
    for model_key in ordered_model_key_list:
        if model_key not in f:
            raise Exception('plot_compare_theta_sequences: model_key: %s not found in summary_data_file_path: %s' %
                            (model_key, summary_data_file_path))
        if not 'decoded_pos_error' in f[model_key]:
            raise Exception('plot_compare_theta_sequences: decoded_pos_error feature for model_key: %s not found in '
                            'summary_data_file_path: %s' %
                            (model_key, summary_data_file_path))
        for pop_name in f[model_key]['decoded_pos_error']:
            instance_index = instance_index_dict[model_key]
            decoded_pos_error_dict[model_key][pop_name].extend(
                np.mean(f[model_key]['decoded_pos_error'][pop_name][str(instance_index)][:], axis=1))

        if not 'theta_sequence_score' in f[model_key]:
            raise Exception('plot_compare_theta_sequences: theta_sequence_score feature for model_key: %s not found in '
                            'summary_data_file_path: %s' %
                            (model_key, summary_data_file_path))
        for pop_name in f[model_key]['theta_sequence_score']:
            instance_index = instance_index_dict[model_key]
            theta_sequence_score_dict[model_key][pop_name].extend(
                f[model_key]['theta_sequence_score'][pop_name][str(instance_index)][:])




fig, flat_axes = plt.subplots(2, 1, figsize=(2.7 * len(pop_order), 5.4))

lines = []
handles = []
for pop_name in pop_order:
    lines.append(Line2D([0], [0], color=color_dict[pop_name]))
    handles.append(pop_name)

axis = 0
pos_start = 0
xlabels = []
for model_key in ordered_model_key_list:
    model_label = model_label_dict[model_key]
    xlabels.append(model_label)
    for pop_name in pop_order:
        bp = flat_axes[axis].boxplot(decoded_pos_error_dict[model_key][pop_name],
                                     positions=[pos_start], patch_artist=True, showfliers=False, widths=0.75)
        if color_dict[pop_name] is not None:
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=color_dict[pop_name])
        for patch in bp['boxes']:
            patch.set(facecolor='white')
        pos_start += 1
    pos_start += 1
flat_axes[axis].set_xticks([1, 5, 9])
flat_axes[axis].set_xticklabels(xlabels)
flat_axes[axis].set_ylabel('Fraction of track')
flat_axes[axis].set_ylim(0., flat_axes[axis].get_ylim()[1] * 1.1)
flat_axes[axis].set_title('Decoded position error', y=1.05, fontsize=mpl.rcParams['font.size'])
flat_axes[axis].legend(lines, handles, loc='best', frameon=False, framealpha=0.5, handlelength=1)

axis += 1
pos_start = 0
xlabels = []
for model_key in ordered_model_key_list:
    model_label = model_label_dict[model_key]
    xlabels.append(model_label)
    for pop_name in pop_order:
        bp = flat_axes[axis].boxplot(theta_sequence_score_dict[model_key][pop_name],
                                     positions=[pos_start], patch_artist=True, showfliers=False, widths=0.75)
        if color_dict[pop_name] is not None:
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=color_dict[pop_name])
        for patch in bp['boxes']:
            patch.set(facecolor='white')
        pos_start += 1
    pos_start += 1
flat_axes[axis].set_xticks([1, 5, 9])
flat_axes[axis].set_ylabel('Explained variance')
flat_axes[axis].set_ylim(0., flat_axes[axis].get_ylim()[1] * 1.1)
flat_axes[axis].set_title('Theta sequence score', y=1.05, fontsize=mpl.rcParams['font.size'])
flat_axes[axis].set_xticklabels(xlabels)

clean_axes(flat_axes)
fig.tight_layout()
fig.subplots_adjust(hspace=0.8)

plt.show()
