import click
from nested.utils import read_from_yaml, Context
from simple_network_analysis_utils import *

context = Context()


@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              default='data')
@click.option("--export-data-file-path", type=click.Path(exists=False, file_okay=True, dir_okay=False), required=False,
              default=None)
@click.option("--export-data-key", type=str, default='0')
@click.option("--model-key", type=str, default='J')
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--export", is_flag=True)
@click.option("--interactive", is_flag=True)
def main(config_file_path, data_dir, export_data_file_path, export_data_key, model_key, decode_window_dur, export,
         interactive):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_file_path: str (path to .hdf5 file)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param model_key: str; key to label dataset exported to .hdf5 file
    :param decode_window_dur: float (ms)
    :param export: bool
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = read_from_yaml(config_file_path)
    context.update(kwargs)

    current_time = time.time()

    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'
    processed_group_key = 'simple_network_processed_data'

    decoded_pos_matrix_dict_list = []

    first = True
    for data_file_name in context.data_file_name_list:
        data_file_path = data_dir + '/' + data_file_name
        if not os.path.isfile(data_file_path):
            raise IOError('export_simple_network_decoded_run_instances: invalid data_file_path: %s' %
                          data_file_path)
        with h5py.File(data_file_path, 'r') as f:
            if first:
                group = get_h5py_group(f, [export_data_key, group_key])
                subgroup = get_h5py_group(group, [shared_context_key])
                decode_duration = get_h5py_attr(subgroup.attrs, 'duration')
                actual_position = np.arange(0., decode_duration, decode_window_dur) + \
                                  decode_window_dur / 2.
                first = False

            group = get_h5py_group(f, [context.export_data_key])
            if processed_group_key not in group or shared_context_key not in group[processed_group_key] or \
                    'decoded_pos_matrix' not in group[processed_group_key][shared_context_key]:
                raise RuntimeError('export_simple_network_decoded_run_instances: data_file_path: %s does not'
                                   'contain required decoded_pos_matrix' % data_file_path)
        decoded_pos_matrix_dict = load_decoded_data(data_file_path, export_data_key)
        decoded_pos_matrix_dict_list.append(decoded_pos_matrix_dict)

    decoded_pos_error_list_dict = {}
    sequence_len_list_dict = {}

    for decoded_pos_matrix_dict in decoded_pos_matrix_dict_list:
        for pop_name in decoded_pos_matrix_dict:
            if pop_name not in decoded_pos_error_list_dict:
                decoded_pos_error_list_dict[pop_name] = []
                sequence_len_list_dict[pop_name] = []
            this_decoded_pos_matrix = decoded_pos_matrix_dict[pop_name][:, :] / decode_duration
            num_trials = this_decoded_pos_matrix.shape[0]
            this_decoded_pos_error_list = []
            this_sequence_len_list = []
            for trial in range(num_trials):
                this_trial_pos = this_decoded_pos_matrix[trial, :]
                this_trial_error = np.subtract(actual_position / decode_duration, this_trial_pos)
                if np.all(np.isnan(this_trial_error)):
                    continue
                for i in range(len(this_trial_error)):
                    if np.isnan(this_trial_error[i]):
                        if i == 0:
                            j = np.where(~np.isnan(this_trial_error))[0][0]
                        else:
                            j = np.where(~np.isnan(this_trial_error[:i]))[0][-1]
                        this_trial_error[i] = this_trial_error[j]
                this_trial_error[np.where(this_trial_error < -0.5)] += 1.
                this_trial_error[np.where(this_trial_error > 0.5)] -= 1.
                analytic_signal = hilbert(this_trial_error)
                amplitude_envelope = np.abs(analytic_signal)

                this_decoded_pos_error_list.append(np.abs(this_trial_error))
                this_sequence_len_list.append(2. * amplitude_envelope)
            decoded_pos_error_list_dict[pop_name].append(this_decoded_pos_error_list)
            sequence_len_list_dict[pop_name].append(this_sequence_len_list)
    if export:
        if export_data_file_path is None or not os.path.isfile(export_data_file_path):
            raise IOError('export_simple_network_decoded_run_instances: invalid export_data_file_path: %s' %
                          export_data_file_path)
        with h5py.File(export_data_file_path, 'a') as f:
            group = get_h5py_group(f, [model_key], create=True)
            subgroup = group.create_group('decoded_pos_error')
            for pop_name in decoded_pos_error_list_dict:
                subgroup.create_group(pop_name)
                for i, instance in enumerate(decoded_pos_error_list_dict[pop_name]):
                    subgroup[pop_name].create_dataset(str(i), data=np.array(instance))
            subgroup = group.create_group('theta_sequence_len')
            for pop_name in sequence_len_list_dict:
                subgroup.create_group(pop_name)
                for i, instance in enumerate(sequence_len_list_dict[pop_name]):
                    subgroup[pop_name].create_dataset(str(i), data=np.array(instance))

    if interactive:
        context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
