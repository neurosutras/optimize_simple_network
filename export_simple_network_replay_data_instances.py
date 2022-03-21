import click
from nested.optimize_utils import Context, read_from_yaml
from simple_network_analysis_utils import *
from analyze_simple_network_replay_rhythmicity import load_replay_fft_trial_matrix_from_file

context = Context()

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              default='data')
@click.option("--export-data-file-path", type=click.Path(exists=False, file_okay=True, dir_okay=False), required=False,
              default=None)
@click.option("--export-data-key", type=str, default='0')
@click.option("--model-key", type=str, default='J')
@click.option("--template-group-key", type=str, default='simple_network_exported_run_data')
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--export", is_flag=True)
@click.option("--interactive", is_flag=True)
def main(config_file_path, data_dir, export_data_file_path, export_data_key, model_key, template_group_key,
         decode_window_dur, export, interactive):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_file_path: str (path to .hdf5 file)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param model_key: str; key to label dataset exported to .hdf5 file
    :param template_group_key: str
    :param decode_window_dur: float (ms)
    :param export: bool
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())

    if config_file_path is None or not os.path.isfile(config_file_path):
        raise RuntimeError('export_simple_network_replay_data_instances: invalid config_file_path: %s' %
                           config_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(config_dict)

    if 'replay_data_file_name_list' not in context():
        raise RuntimeError('export_simple_network_replay_data_instances: config_file_path: %s does not'
                           'contain required replay_data_file_name_list' % config_file_path)
    if 'template_data_file_name_list' not in context():
        raise RuntimeError('export_simple_network_replay_data_instances: config_file_path: %s does not'
                           'contain required template_data_file_name_list' % config_file_path)

    template_data_file_path = data_dir + '/' + context.template_data_file_name_list[0]
    if not os.path.isfile(template_data_file_path):
        raise IOError('export_simple_network_replay_data_instances: invalid template_data_file_path: %s' %
                      template_data_file_path)

    shared_context_key = 'shared_context'
    with h5py.File(template_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, template_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        template_duration = get_h5py_attr(subgroup.attrs, 'duration')

    decoded_pos_matrix_dict_list = []
    fft_power_matrix_dict_list = []
    for replay_data_file_name in context.replay_data_file_name_list:
        replay_data_file_path = data_dir + '/' + replay_data_file_name
        if not os.path.isfile(replay_data_file_path):
            raise IOError('export_simple_network_replay_data_instances: invalid replay_data_file_path: %s' %
                          replay_data_file_path)
        decoded_pos_matrix_dict = load_decoded_data(replay_data_file_path, export_data_key)
        decoded_pos_matrix_dict_list.append(decoded_pos_matrix_dict)

        fft_f, fft_power_matrix_dict = \
            load_replay_fft_trial_matrix_from_file(replay_data_file_path, export_data_key)
        fft_power_matrix_dict_list.append(fft_power_matrix_dict)

    fft_power_instance_dict = {}
    for fft_power_matrix_dict in fft_power_matrix_dict_list:
        for pop_name in fft_power_matrix_dict:
            if pop_name not in fft_power_instance_dict:
                fft_power_instance_dict[pop_name] = []
            fft_power_instance_dict[pop_name].append(np.mean(fft_power_matrix_dict[pop_name], axis=0))

    decoded_pos_list_dict = {}
    decoded_velocity_mean_list_dict = {}
    decoded_path_len_list_dict = {}
    decoded_max_step_list_dict = {}

    for decoded_pos_matrix_dict in decoded_pos_matrix_dict_list:
        for pop_name in decoded_pos_matrix_dict:
            if pop_name not in decoded_pos_list_dict:
                decoded_pos_list_dict[pop_name] = []
                decoded_velocity_mean_list_dict[pop_name] = []
                decoded_path_len_list_dict[pop_name] = []
                decoded_max_step_list_dict[pop_name] = []
            this_decoded_pos_matrix = decoded_pos_matrix_dict[pop_name][:, :] / template_duration
            trial_dur = this_decoded_pos_matrix.shape[1] * decode_window_dur / 1000.
            clean_indexes = ~np.isnan(this_decoded_pos_matrix)
            decoded_pos_list_dict[pop_name].append(this_decoded_pos_matrix[clean_indexes])
            this_velocity_mean_list = []
            this_path_len_list = []
            this_max_step_list = []
            for trial in range(this_decoded_pos_matrix.shape[0]):
                this_trial_pos = this_decoded_pos_matrix[trial, :]
                clean_indexes = ~np.isnan(this_trial_pos)
                if len(clean_indexes) > 0:
                    this_trial_diff = np.diff(this_trial_pos[clean_indexes])
                    this_trial_diff[np.where(this_trial_diff < -0.5)] += 1.
                    this_trial_diff[np.where(this_trial_diff > 0.5)] -= 1.
                    this_path_len = np.sum(np.abs(this_trial_diff))
                    this_path_len_list.append(this_path_len)
                    this_trial_velocity = np.sum(this_trial_diff) / trial_dur
                    this_velocity_mean_list.append(this_trial_velocity)
                if len(this_trial_diff) > 0:
                    this_max_step_list.append(np.max(np.abs(this_trial_diff)))
            decoded_path_len_list_dict[pop_name].append(this_path_len_list)
            decoded_velocity_mean_list_dict[pop_name].append(this_velocity_mean_list)
            decoded_max_step_list_dict[pop_name].append(this_max_step_list)

    if export:
        if export_data_file_path is None or not os.path.isfile(export_data_file_path):
            raise IOError('export_simple_network_replay_data_instances: invalid export_data_file_path: %s' %
                          export_data_file_path)
        with h5py.File(export_data_file_path, 'a') as f:
            group = get_h5py_group(f, ['shared_context'], create=True)
            if 'replay_fft_f' not in group:
                group.create_dataset('replay_fft_f', data=fft_f)

            group = get_h5py_group(f, [model_key], create=True)
            if 'replay_fft_power' not in group:
                subgroup = group.create_group('replay_fft_power')
                for pop_name in fft_power_instance_dict:
                    subgroup.create_dataset(pop_name, data=np.array(fft_power_instance_dict[pop_name]))
            if 'replay_decoded_pos' not in group:
                subgroup = group.create_group('replay_decoded_pos')
                for pop_name in decoded_pos_list_dict:
                    subgroup.create_group(pop_name)
                    for i, instance in enumerate(decoded_pos_list_dict[pop_name]):
                        subgroup[pop_name].create_dataset(str(i), data=np.array(instance))
            if 'replay_decoded_path_len' not in group:
                subgroup = group.create_group('replay_decoded_path_len')
                for pop_name in decoded_path_len_list_dict:
                    subgroup.create_group(pop_name)
                    for i, instance in enumerate(decoded_path_len_list_dict[pop_name]):
                        subgroup[pop_name].create_dataset(str(i), data=np.array(instance))
            if 'replay_decoded_velocity' not in group:
                subgroup = group.create_group('replay_decoded_velocity')
                for pop_name in decoded_velocity_mean_list_dict:
                    subgroup.create_group(pop_name)
                    for i, instance in enumerate(decoded_velocity_mean_list_dict[pop_name]):
                        subgroup[pop_name].create_dataset(str(i), data=np.array(instance))
            if 'replay_max_step' not in group:
                subgroup = group.create_group('replay_max_step')
                for pop_name in decoded_max_step_list_dict:
                    subgroup.create_group(pop_name)
                    for i, instance in enumerate(decoded_max_step_list_dict[pop_name]):
                        subgroup[pop_name].create_dataset(str(i), data=np.array(instance))

    if interactive:
        context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
