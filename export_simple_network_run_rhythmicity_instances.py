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
@click.option("--fine-binned-dt", type=float, default=1.)
@click.option("--export", is_flag=True)
@click.option("--interactive", is_flag=True)
def main(config_file_path, data_dir, export_data_file_path, export_data_key, model_key, fine_binned_dt, export,
         interactive):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_file_path: str (path to .hdf5 file)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param model_key: str; key to label dataset exported to .hdf5 file
    :param fine_binned_dt: float (ms)
    :param export: bool
    :param interactive: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = read_from_yaml(config_file_path)
    context.update(kwargs)

    current_time = time.time()

    full_spike_times_dict_list = []
    filter_bands = dict()
    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'

    for data_file_name in context.template_data_file_name_list:
        data_file_path = data_dir + '/' + data_file_name
        if not os.path.isfile(data_file_path):
            raise IOError(
                'export_simple_network_run_rhythmicity_instances: invalid data file path: %s' % data_file_path)
        with h5py.File(data_file_path, 'r') as f:
            group = get_h5py_group(f, [export_data_key, group_key])
            subgroup = group[shared_context_key]
            duration = get_h5py_attr(subgroup.attrs, 'duration')
            buffer = get_h5py_attr(subgroup.attrs, 'buffer')

            data_group = subgroup['filter_bands']
            for this_filter in data_group:
                filter_bands[this_filter] = data_group[this_filter][:]

            for trial_key in (key for key in group if key != shared_context_key):
                subgroup = group[trial_key]
                data_group = subgroup['full_spike_times']
                full_spike_times_dict = defaultdict(dict)
                for pop_name in data_group:
                    for gid_key in data_group[pop_name]:
                        full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]
                full_spike_times_dict_list.append(full_spike_times_dict)

    buffered_binned_t_edges = \
        np.arange(-buffer, duration + buffer + fine_binned_dt / 2., fine_binned_dt)
    buffered_binned_t = buffered_binned_t_edges[:-1] + fine_binned_dt / 2.
    fine_binned_t_edges = np.arange(0., duration + fine_binned_dt / 2., fine_binned_dt)
    fine_binned_t = fine_binned_t_edges[:-1] + fine_binned_dt / 2.

    fft_power_list_dict = {}
    fft_power_nested_gamma_list_dict = {}

    for i, full_spike_times_dict in enumerate(full_spike_times_dict_list):
        current_time = time.time()
        buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, buffered_binned_t_edges)
        buffered_pop_mean_rate_from_binned_spike_count_dict = \
            get_pop_mean_rate_from_binned_spike_count(buffered_binned_spike_count_dict, dt=fine_binned_dt)

        fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
        filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(buffered_pop_mean_rate_from_binned_spike_count_dict,
                                                   filter_bands, input_t=buffered_binned_t,
                                                   valid_t=buffered_binned_t, output_t=fine_binned_t, pad=True)

        if 'Gamma' in filter_envelope_dict:
            fft_f_nested_gamma_dict, fft_power_nested_gamma_dict = \
                get_pop_bandpass_envelope_fft(filter_envelope_dict['Gamma'], dt=fine_binned_dt)

        for pop_name in fft_power_dict:
            if pop_name not in fft_power_list_dict:
                fft_power_list_dict[pop_name] = []
                fft_power_nested_gamma_list_dict[pop_name] = []
            fft_power_list_dict[pop_name].append(fft_power_dict[pop_name])
            if 'Gamma' in filter_envelope_dict:
                fft_power_nested_gamma_list_dict[pop_name].append(fft_power_nested_gamma_dict[pop_name])

        print('export_simple_network_run_rhythmicity_instances took %.1f s to process data from %i/%i trials from '
              '%i network instances' % (time.time() - current_time, (i + 1), len(full_spike_times_dict_list),
                                        len(context.template_data_file_name_list)))
        sys.stdout.flush()

    if export:
        if export_data_file_path is None or not os.path.isfile(export_data_file_path):
            raise IOError('export_simple_network_run_rhythmicity_instances: invalid export_data_file_path: %s' %
                          export_data_file_path)
        with h5py.File(export_data_file_path, 'a') as f:
            group = get_h5py_group(f, [model_key], create=True)
            subgroup = group.create_group('fft_power')
            for pop_name in fft_power_list_dict:
                subgroup.create_dataset(pop_name, data=np.array(fft_power_list_dict[pop_name]).flatten())
            subgroup = group.create_group('fft_power_nested_gamma')
            for pop_name in fft_power_nested_gamma_list_dict:
                subgroup.create_dataset(pop_name, data=np.array(fft_power_nested_gamma_list_dict[pop_name]).flatten())

    if interactive:
        context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
