from nested.parallel import *
from nested.optimize_utils import *
from simple_network_analysis_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--replay-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--group-key", type=str, default='simple_network_exported_replay_data')
@click.option("--export-data-key", type=int, default=0)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False,
              default=None)
@click.option("--binned_dt", type=float, default=1.)
@click.option("--output-dir", type=str, default='data')
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--export", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, replay_data_file_path, group_key, export_data_key, config_file_path, binned_dt, output_dir, interactive,
         verbose, export, plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param replay_data_file_path: str (path)
    :param group_key: str
    :param export_data_key: str
    :param config_file_path: str (path); contains plot settings
    :param binned_dt: float
    :param output_dir: str (path to dir)
    :param interactive: bool
    :param verbose: int
    :param export: bool
    :param plot: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()

    config_parallel_interface(__file__, config_file_path=config_file_path, disp=context.disp,
                              interface=context.interface, output_dir=output_dir, verbose=verbose, plot=plot,
                              debug=debug, export_data_key=export_data_key, export=export,
                              group_key=group_key, **kwargs)

    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None

    current_time = time.time()

    if not os.path.isfile(replay_data_file_path):
        raise IOError('analyze_simple_network_replay_rhythmicity: invalid replay_data_file_path: %s' %
                      replay_data_file_path)

    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(context.replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, group_key])
        trial_keys = [key for key in group if key != shared_context_key]

        group = get_h5py_group(f, [export_data_key])
        if processed_group_key in group and shared_context_key in group[processed_group_key] and \
                'fft_power_matrix' in group[processed_group_key][shared_context_key]:
            replay_data_is_processed = True
        else:
            replay_data_is_processed = False

    if not replay_data_is_processed:
        fft_f, fft_power_matrix_dict = \
            compute_replay_fft_trial_matrix(replay_data_file_path, trial_keys, group_key, export_data_key,  binned_dt,
                                            export, context.disp)
    else:
        fft_f, fft_power_matrix_dict = \
            load_replay_fft_trial_matrix_from_file(replay_data_file_path, export_data_key)

    if context.disp:
        print('analyze_simple_network_replay_rhythmicity: analyzing rhythmicity for %i replay trials from file: %s '
              'took %.1f s' % (len(trial_keys), replay_data_file_path, time.time() - current_time))
        sys.stdout.flush()

    fft_power_mean_dict = {}
    fft_power_sem_dict = {}
    for pop_name in fft_power_matrix_dict:
        fft_power_sem_dict[pop_name] = np.std(fft_power_matrix_dict[pop_name], axis=0) / \
                                       np.sqrt(fft_power_matrix_dict[pop_name].shape[0])
        fft_power_mean_dict[pop_name] = np.mean(fft_power_matrix_dict[pop_name], axis=0)

    if plot:
        plot_rhythmicity_psd(fft_f, fft_power_mean_dict, fft_power_sem_dict, pop_order=context.pop_order,
                             label_dict=context.label_dict, color_dict=context.color_dict)
        plt.show()

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


def compute_replay_fft_single_trial(replay_data_file_path, trial_key, group_key, export_data_key, binned_dt,
                                    disp=False):
    """

    :param replay_data_file_path: str (path)
    :param trial_key: str
    :param group_key: str
    :param export_data_key: str
    :param binned_dt: float (ms)
    :param disp: bool
    """

    full_spike_times_dict = defaultdict(dict)
    filter_bands = dict()

    shared_context_key = 'shared_context'
    with h5py.File(replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, group_key])
        subgroup = group[shared_context_key]
        duration = get_h5py_attr(subgroup.attrs, 'duration')
        buffer = get_h5py_attr(subgroup.attrs, 'buffer')

        data_group = subgroup['filter_bands']
        for this_filter in data_group:
            filter_bands[this_filter] = data_group[this_filter][:]

        if trial_key not in group:
            raise RuntimeError('compute_replay_fft_single_trial: data for trial: %i not found in data '
                               'file path: %s' % (trial_key, replay_data_file_path))
        subgroup = get_h5py_group(group, [trial_key])
        data_group = subgroup['full_spike_times']
        for pop_name in data_group:
            for gid_key in data_group[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]

    fine_buffered_binned_t_edges = \
        np.arange(-buffer, duration + buffer + binned_dt / 2., binned_dt)
    fine_buffered_binned_t = fine_buffered_binned_t_edges[:-1] + binned_dt / 2.
    fine_binned_t_edges = np.arange(0., duration + binned_dt / 2., binned_dt)
    fine_binned_t = fine_binned_t_edges[:-1] + binned_dt / 2.

    fine_buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict,
                                                                        fine_buffered_binned_t_edges)
    fine_buffered_pop_mean_rate_from_binned_spike_count_dict = \
        get_pop_mean_rate_from_binned_spike_count(fine_buffered_binned_spike_count_dict, dt=binned_dt)

    fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
    filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
        get_pop_bandpass_filtered_signal_stats(fine_buffered_pop_mean_rate_from_binned_spike_count_dict,
                                               filter_bands, input_t=fine_buffered_binned_t,
                                               valid_t=fine_buffered_binned_t, output_t=fine_binned_t, pad=False,
                                               plot=False)
    fft_f = next(iter(fft_f_dict.values()))

    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(context.temp_output_path, 'a') as f:
        group = get_h5py_group(f, [export_data_key, processed_group_key, trial_key], create=True)
        subgroup = group.create_group('fft_power')
        for pop_name in fft_power_dict:
            subgroup.create_dataset(pop_name, data=fft_power_dict[pop_name], compression='gzip')
        group = get_h5py_group(f, [export_data_key, processed_group_key, shared_context_key], create=True)
        if 'fft_f' not in group:
            group.create_dataset('fft_f', data=fft_f, compression='gzip')

    if disp:
        print('compute_replay_fft_single_trial: pid: %i exported data for trial: %s to temp_output_path: %s' %
              (os.getpid(), trial_key, context.temp_output_path))
        sys.stdout.flush()


def compute_replay_fft_trial_matrix(replay_data_file_path, trial_keys, group_key, export_data_key, binned_dt, export,
                                    disp=False):
    """

    :param replay_data_file_path: str (path)
    :param trial_keys: list of str
    :param group_key: str
    :param export_data_key: str
    :param binned_dt: float (ms)
    :param export: bool
    :param disp: bool
    :return: tuple: array of float (num_freq_bins), dict of 2d array of float (num_trials, num_freq_bins)
    """
    num_trials = len(trial_keys)
    sequences = [[replay_data_file_path] * num_trials, trial_keys, [group_key] * num_trials,
                 [export_data_key] * num_trials, [binned_dt] * num_trials, [disp] * num_trials]
    context.interface.map(compute_replay_fft_single_trial, *sequences)

    fft_power_list_dict = dict()
    fft_power_matrix_dict = dict()

    start_time = time.time()
    temp_output_path_list = \
        [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
         if os.path.isfile(temp_output_path)]
    trial_key_list = []

    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    first = True
    for temp_output_path in temp_output_path_list:
        with h5py.File(temp_output_path, 'r') as f:
            group = get_h5py_group(f, [export_data_key, processed_group_key])
            for trial_key in (key for key in group if key != shared_context_key):
                trial_key_list.append(trial_key)
                subgroup = get_h5py_group(group, [trial_key])
                data_group = subgroup['fft_power']
                for pop_name in data_group:
                    this_fft_power = data_group[pop_name][:]
                    if pop_name not in fft_power_list_dict:
                        fft_power_list_dict[pop_name] = []
                    fft_power_list_dict[pop_name].append(this_fft_power)
            if first:
                subgroup = get_h5py_group(group, [shared_context_key])
                fft_f = subgroup['fft_f'][:]
                first = False

    sorted_trial_indexes = np.argsort(np.array(trial_key_list, dtype=int))
    for pop_name in fft_power_list_dict:
        fft_power_matrix_dict[pop_name] = \
            np.asarray(fft_power_list_dict[pop_name])[sorted_trial_indexes,:]

    if export:
        with h5py.File(replay_data_file_path, 'a') as f:
            group = get_h5py_group(f, [export_data_key, processed_group_key, shared_context_key], create=True)
            group.create_dataset('fft_f', data=fft_f, compression='gzip')
            subgroup = group.create_group('fft_power_matrix')
            for pop_name in fft_power_matrix_dict:
                subgroup.create_dataset(pop_name, data=fft_power_matrix_dict[pop_name], compression='gzip')
        if disp:
            print('compute_replay_fft_trial_matrix: exporting to replay_data_file_path: %s took %.1f s' %
                  (replay_data_file_path, time.time() - start_time))

    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)

    return fft_f, fft_power_matrix_dict


def load_replay_fft_trial_matrix_from_file(replay_data_file_path, export_data_key):
    """

    :param replay_data_file_path: str (path)
    :param export_data_key: str
    :return: tuple: array of float (num_freq_bins), dict of 2d array of float (num_trials, num_freq_bins)
    """
    fft_power_matrix_dict = {}

    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, processed_group_key, shared_context_key])
        fft_f = group['fft_f'][:]
        subgroup = group['fft_power_matrix']
        for pop_name in subgroup:
            fft_power_matrix_dict[pop_name] = subgroup[pop_name][:]

    return fft_f, fft_power_matrix_dict


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
