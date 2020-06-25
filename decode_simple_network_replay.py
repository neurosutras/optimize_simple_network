from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--run-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--replay-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--plot-n-trials", type=int, default=10)
@click.option("--window-dur", type=float, default=20.)
@click.option("--step-dur", type=float, default=20.)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, run_data_file_path, replay_data_file_path, plot_n_trials, window_dur, step_dur, interactive, verbose,
         plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param run_data_file_path: str (path)
    :param replay_data_file_path: str (path)
    :param plot_n_trials: int
    :param window_dur: float
    :param step_dur: float
    :param interactive: bool
    :param verbose: int
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

    config_parallel_interface(__file__, disp=context.disp, interface=context.interface, verbose=verbose, plot=plot,
                              debug=debug, **kwargs)

    if context.debug:
        return

    start_time = time.time()

    if not context.run_data_is_processed:
        append_processed_run_data_to_file()
    context.interface.apply(load_processed_run_data_from_file)

    if context.disp:
        print('decode_simple_network_replay: processing run data for %i trials took %.1f s' %
              (len(context.run_trial_keys), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    context.interface.apply(update_worker_contexts, replay_binned_t=context.replay_binned_t,
                            plot_replay_trial_keys=context.plot_replay_trial_keys)
    context.interface.apply(init_context)

    if not context.replay_data_is_processed:
        num_replay_trials = len(context.replay_trial_keys)
        sequences = [context.replay_trial_keys, [True] * num_replay_trials]
        context.interface.map(process_replay_trial_data, *sequences)
        append_processed_replay_data_to_file()
    elif plot:
        sequences = [context.plot_replay_trial_keys, [False] * context.plot_n_trials]
        context.interface.map(process_replay_trial_data, *sequences)

    if plot:
        analyze_decoded_position_replay_from_file(context.replay_data_file_path,
                                                  (context.run_binned_t[0], context.run_binned_t[-1]), plot=True)

    if context.disp:
        print('decode_simple_network_replay: processing replay data for %i trials took %.1f s' %
              (len(context.replay_trial_keys), time.time() - start_time))
        sys.stdout.flush()

    if context.plot:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


def config_controller():

    if not os.path.isfile(context.run_data_file_path):
        raise IOError('decode_simple_network_replay: invalid run_data_file_path: %s' %
                      context.run_data_file_path)
    with h5py.File(context.run_data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context'])
        if 'duration' in group.attrs:
            baks_pad_dur = group.attrs['duration']
        if 'trial_averaged_firing_rate_matrix' in group:
            run_data_is_processed = True
        else:
            run_data_is_processed = False
        run_binned_t = group['binned_t'][:]
        run_data_group_key = 'simple_network_exported_data'
        run_trial_keys = [key for key in f if run_data_group_key in f[key]]

    if not os.path.isfile(context.replay_data_file_path):
        raise IOError('decode_simple_network_replay: invalid replay_data_file_path: %s' %
                      context.replay_data_file_path)
    with h5py.File(context.replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context'])
        replay_binned_t = group['buffered_binned_t'][:]
        replay_data_group_key = 'simple_network_exported_data'
        replay_trial_keys = [key for key in f if replay_data_group_key in f[key]]
        context.plot_n_trials = min(int(context.plot_n_trials), len(replay_trial_keys))
        plot_replay_trial_keys = \
            list(np.random.choice(replay_trial_keys, context.plot_n_trials, replace=False))
        if 'decoded_pos_matrix' in group:
            replay_data_is_processed = True
        else:
            replay_data_is_processed = False

    context.update(locals())


def config_worker():

    baks_alpha = 4.7725100028345535
    baks_beta = 0.41969058927343522
    baks_pad_dur = 3000.  # ms
    baks_wrap_around = True

    context.update(locals())


def init_context():

    replay_binned_t = context.replay_binned_t
    replay_binned_dt = replay_binned_t[1] - replay_binned_t[0]
    window_dur = context.window_dur
    step_dur = context.step_dur

    align_to_t = 0.
    half_window_bins = int(window_dur // replay_binned_dt // 2)
    window_bins = int(2 * half_window_bins + 1)
    window_dur = window_bins * replay_binned_dt

    step_bins = step_dur // replay_binned_dt
    step_dur = step_bins * replay_binned_dt
    half_step_dur = step_dur / 2.

    # if possible, include a bin starting at time zero.
    binned_t_center_indexes = []
    this_center_index = np.where(replay_binned_t >= align_to_t)[0] + half_window_bins
    if len(this_center_index) > 0:
        this_center_index = this_center_index[0]
        if this_center_index < half_window_bins:
            this_center_index = half_window_bins
            binned_t_center_indexes.append(this_center_index)
        else:
            while this_center_index > half_window_bins:
                binned_t_center_indexes.append(this_center_index)
                this_center_index -= step_bins
            binned_t_center_indexes.reverse()
    else:
        this_center_index = half_window_bins
        binned_t_center_indexes.append(this_center_index)
    this_center_index = binned_t_center_indexes[-1] + step_bins
    while this_center_index < len(replay_binned_t) - half_window_bins:
        binned_t_center_indexes.append(this_center_index)
        this_center_index += step_bins

    replay_binned_t_center_indexes = np.array(binned_t_center_indexes, dtype='int')
    decode_binned_t = replay_binned_t[replay_binned_t_center_indexes]

    context.update(locals())


def export_processed_run_trial_data(trial_key):
    """

    :param trial_key: str
    """
    start_time = time.time()

    full_spike_times_dict = dict()
    tuning_peak_locs = dict()
    run_data_group_key = 'simple_network_exported_data'
    with h5py.File(context.run_data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context'])
        if 'tuning_peak_locs' in group and len(group['tuning_peak_locs']) > 0:
            subgroup = group['tuning_peak_locs']
            for pop_name in subgroup:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(subgroup[pop_name]['target_gids'], subgroup[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        run_binned_t = group['binned_t'][:]
        group = get_h5py_group(f, [trial_key, run_data_group_key])
        subgroup = group['full_spike_times']
        for pop_name in subgroup:
            full_spike_times_dict[pop_name] = dict()
            for gid_key in subgroup[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]

    if context.disp:
        print('export_processed_run_trial_data: pid: %i took %.1f s to load spikes times for trial: %s' %
              (os.getpid(), time.time() - start_time, trial_key))
        sys.stdout.flush()
    current_time = time.time()

    binned_firing_rates_dict = \
        infer_firing_rates_baks(full_spike_times_dict, run_binned_t, alpha=context.baks_alpha, beta=context.baks_beta,
                                pad_dur=context.baks_pad_dur, wrap_around=context.baks_wrap_around)

    sorted_gid_dict = dict()
    for pop_name in full_spike_times_dict:
        if pop_name in tuning_peak_locs:
            this_target_gids = np.array(list(tuning_peak_locs[pop_name].keys()))
            this_peak_locs = np.array(list(tuning_peak_locs[pop_name].values()))
            indexes = np.argsort(this_peak_locs)
            sorted_gid_dict[pop_name] = this_target_gids[indexes]
        else:
            this_target_gids = [int(gid_key) for gid_key in full_spike_times_dict[pop_name]]
            sorted_gid_dict[pop_name] = np.array(sorted(this_target_gids), dtype='int')

    firing_rates_matrix_dict = dict()
    for pop_name in binned_firing_rates_dict:
        firing_rates_matrix_dict[pop_name] = np.empty((len(binned_firing_rates_dict[pop_name]), len(run_binned_t)))
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            firing_rates_matrix_dict[pop_name][i, :] = binned_firing_rates_dict[pop_name][gid]

    if context.disp:
        print('export_processed_run_trial_data: pid: %i took %.1f s to process firing rates for trial: %s' %
              (os.getpid(), time.time() - current_time, trial_key))
        sys.stdout.flush()

    with h5py.File(context.temp_output_path, 'a') as f:
        if 'shared_context' not in f:
            group = get_h5py_group(f, ['shared_context'], create=True)
            subgroup = group.create_group('sorted_gids')
            for pop_name in sorted_gid_dict:
                subgroup.create_dataset(pop_name, data=sorted_gid_dict[pop_name], compression='gzip')
        exported_data_key = 'simple_network_processed_data'
        group = get_h5py_group(f, [trial_key, exported_data_key], create=True)
        subgroup = group.create_group('firing_rates_matrix')
        for pop_name in firing_rates_matrix_dict:
            subgroup.create_dataset(pop_name, data=firing_rates_matrix_dict[pop_name], compression='gzip')

    if context.disp:
        print('export_processed_run_trial_data: pid: %i exported data for trial: %s to temp_output_path: %s' %
              (os.getpid(), trial_key, context.temp_output_path))
        sys.stdout.flush()


def append_processed_run_data_to_file():

    start_time = time.time()
    context.interface.apply(update_worker_contexts, baks_pad_dur=context.baks_pad_dur)
    context.interface.map(export_processed_run_trial_data, *[context.run_trial_keys])
    temp_output_path_list = [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
                             if os.path.isfile(temp_output_path)]

    run_trial_firing_rates_matrix_list_dict = dict()
    first = True
    for temp_output_path in temp_output_path_list:
        with h5py.File(temp_output_path, 'r') as f:
            if first:
                group = get_h5py_group(f, ['shared_context'])
                sorted_gid_dict = dict()
                subgroup = group['sorted_gids']
                for pop_name in subgroup:
                    sorted_gid_dict[pop_name] = subgroup[pop_name][:]
                first = False
            exported_data_key = 'simple_network_processed_data'
            for trial_key in (key for key in f if exported_data_key in f[key]):
                group = get_h5py_group(f, [trial_key, exported_data_key])
                subgroup = group['firing_rates_matrix']
                for pop_name in subgroup:
                    this_run_trial_firing_rates_matrix = subgroup[pop_name][:]
                    if pop_name not in run_trial_firing_rates_matrix_list_dict:
                        run_trial_firing_rates_matrix_list_dict[pop_name] = []
                    run_trial_firing_rates_matrix_list_dict[pop_name].append(this_run_trial_firing_rates_matrix)

    trial_averaged_run_firing_rate_matrix_dict = dict()
    for pop_name in run_trial_firing_rates_matrix_list_dict:
        trial_averaged_run_firing_rate_matrix_dict[pop_name] = \
            np.mean(run_trial_firing_rates_matrix_list_dict[pop_name], axis=0)

    with h5py.File(context.run_data_file_path, 'a') as f:
        group = get_h5py_group(f, ['shared_context'])
        subgroup = group.create_group('sorted_gids')
        for pop_name in sorted_gid_dict:
            subgroup.create_dataset(pop_name, data=sorted_gid_dict[pop_name], compression='gzip')
        subgroup = group.create_group('trial_averaged_firing_rate_matrix')
        for pop_name in trial_averaged_run_firing_rate_matrix_dict:
            subgroup.create_dataset(pop_name, data=trial_averaged_run_firing_rate_matrix_dict[pop_name],
                                    compression='gzip')

    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)

    if context.disp:
        print('append_processed_run_data_to_file: pid: %i; exporting to run_data_file_path: %s '
              'took %.1f s' % (os.getpid(), context.run_data_file_path, time.time() - start_time))


def load_processed_run_data_from_file():
    """

    """
    sorted_gid_dict = dict()
    run_firing_rate_matrix_dict = dict()
    with h5py.File(context.run_data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context'])
        run_binned_t = group['binned_t'][:]
        subgroup = group['sorted_gids']
        for pop_name in subgroup:
            sorted_gid_dict[pop_name] = subgroup[pop_name][:]
        subgroup = group['trial_averaged_firing_rate_matrix']
        for pop_name in subgroup:
            run_firing_rate_matrix_dict[pop_name] = subgroup[pop_name][:]
    context.update(locals())


def process_replay_trial_data(trial_key, export=True):
    """

    :param trial_key: str
    :param export: bool
    """
    start_time = time.time()

    replay_full_spike_times_dict = dict()
    replay_binned_spike_count_matrix_dict = dict()
    replay_data_group_key = 'simple_network_exported_data'
    with h5py.File(context.replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [trial_key, replay_data_group_key, 'full_spike_times'])
        for pop_name in group:
            if pop_name not in replay_full_spike_times_dict:
                replay_full_spike_times_dict[pop_name] = dict()
            for gid_key in group[pop_name]:
                replay_full_spike_times_dict[pop_name][int(gid_key)] = group[pop_name][gid_key][:]
    replay_full_binned_spike_count_dict = \
        get_binned_spike_count_dict(replay_full_spike_times_dict, context.replay_binned_t)

    for pop_name in replay_full_binned_spike_count_dict:
        replay_binned_spike_count = np.empty((len(replay_full_binned_spike_count_dict[pop_name]),
                                              len(context.replay_binned_t)))
        for i, gid in enumerate(context.sorted_gid_dict[pop_name]):
            replay_binned_spike_count[i, :] = replay_full_binned_spike_count_dict[pop_name][gid]
        replay_binned_spike_count_matrix_dict[pop_name] = replay_binned_spike_count

    p_pos_dict = decode_position_from_offline_replay(context.run_binned_t, context.run_firing_rate_matrix_dict,
                                                     context.replay_binned_t,
                                                     replay_binned_spike_count_matrix_dict,
                                                     context.replay_binned_t_center_indexes,
                                                     context.decode_binned_t, window_dur=context.window_dur)

    decoded_pos_dict = dict()
    for pop_name in p_pos_dict:
        this_decoded_pos = np.empty_like(context.decode_binned_t)
        this_decoded_pos[:] = np.nan
        p_pos = p_pos_dict[pop_name]
        for pos_bin in range(p_pos.shape[1]):
            if np.any(~np.isnan(p_pos[:, pos_bin])):
                index = np.nanargmax(p_pos[:, pos_bin])
                this_decoded_pos[pos_bin] = context.run_binned_t[index]
        decoded_pos_dict[pop_name] = this_decoded_pos

    if context.disp:
        print('process_replay_trial_data: pid: %i took %.1f s to decode position for trial: %s' %
              (os.getpid(), time.time() - start_time, trial_key))
        sys.stdout.flush()

    if export:
        with h5py.File(context.temp_output_path, 'a') as f:
            exported_data_key = 'simple_network_processed_data'
            group = get_h5py_group(f, [trial_key, exported_data_key], create=True)
            subgroup = group.create_group('decoded_position')
            for pop_name in decoded_pos_dict:
                subgroup.create_dataset(pop_name, data=decoded_pos_dict[pop_name], compression='gzip')

        if context.disp:
            print('process_replay_trial_data: pid: %i exported data for trial: %s to temp_output_path: %s' %
                  (os.getpid(), trial_key, context.temp_output_path))
            sys.stdout.flush()

    if context.plot and trial_key in context.plot_replay_trial_keys:
        ordered_pop_names = ['FF', 'E', 'I']
        for pop_name in ordered_pop_names:
            if pop_name not in p_pos_dict:
                ordered_pop_names.remove(pop_name)
        for pop_name in p_pos_dict:
            if pop_name not in ordered_pop_names:
                ordered_pop_names.append(pop_name)
        fig, axes = plt.subplots(2, len(ordered_pop_names), figsize=(3.8 * len(ordered_pop_names) + 0.5, 7.5))
        decoded_x_mesh, decoded_y_mesh = \
            np.meshgrid(context.decode_binned_t - context.half_step_dur, context.run_binned_t)
        for col, pop_name in enumerate(ordered_pop_names):
            p_pos = p_pos_dict[pop_name]
            replay_binned_spike_count = replay_binned_spike_count_matrix_dict[pop_name]
            spikes_x_mesh, spikes_y_mesh = \
                np.meshgrid(context.replay_binned_t, list(range(replay_binned_spike_count.shape[0])))
            axes[1][col].pcolormesh(decoded_x_mesh, decoded_y_mesh, p_pos, vmin=0.)
            axes[1][col].set_xlabel('Time (ms)')
            axes[1][col].set_ylim([context.run_binned_t[-1], context.run_binned_t[0]])
            axes[1][col].set_xlim([context.replay_binned_t[0], context.replay_binned_t[-1]])
            axes[1][col].set_ylabel('Decoded position')
            axes[1][col].set_title('Population: %s' % pop_name)
            axes[0][col].scatter(spikes_x_mesh, spikes_y_mesh, replay_binned_spike_count, c='k')
            axes[0][col].set_xlabel('Time (ms)')
            axes[0][col].set_ylim([replay_binned_spike_count.shape[0] - 1, 0])
            axes[0][col].set_xlim([context.replay_binned_t[0], context.replay_binned_t[-1]])
            axes[0][col].set_ylabel('Sorted cells')
            axes[0][col].set_title('Population: %s' % pop_name)
        fig.suptitle('Trial # %s' % trial_key, y=0.99)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9)


def append_processed_replay_data_to_file():

    start_time = time.time()
    temp_output_path_list = [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
                             if os.path.isfile(temp_output_path)]

    decoded_pos_list_dict = dict()
    for temp_output_path in temp_output_path_list:
        with h5py.File(temp_output_path, 'r') as f:
            exported_data_key = 'simple_network_processed_data'
            for trial_key in (key for key in f if exported_data_key in f[key]):
                group = get_h5py_group(f, [trial_key, exported_data_key])
                subgroup = group['decoded_position']
                for pop_name in subgroup:
                    this_decoded_position = subgroup[pop_name][:]
                    if pop_name not in decoded_pos_list_dict:
                        decoded_pos_list_dict[pop_name] = []
                    decoded_pos_list_dict[pop_name].append(this_decoded_position)

    if context.debug:
        context.update(decoded_pos_list_dict=decoded_pos_list_dict)

    with h5py.File(context.replay_data_file_path, 'a') as f:
        group = get_h5py_group(f, ['shared_context'])
        group.create_dataset('decode_binned_t', data=context.decode_binned_t, compression='gzip')
        subgroup = group.create_group('decoded_pos_matrix')
        for pop_name in decoded_pos_list_dict:
            subgroup.create_dataset(pop_name, data=np.asarray(decoded_pos_list_dict[pop_name]), compression='gzip')

    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)

    if context.disp:
        print('append_processed_replay_data_to_file: pid: %i; exporting to replay_data_file_path: %s '
              'took %.1f s' % (os.getpid(), context.replay_data_file_path, time.time() - start_time))


def analyze_decoded_position_replay_from_file(replay_data_file_path, run_range, plot=True, full_output=False):
    """

    :param replay_data_file_path: str (path)
    :param run_range: tuple of float
    :param plot: bool
    :param full_output: bool
    :return: tuple
    """
    track_start = run_range[0]
    track_length = run_range[1] - run_range[0]

    decoded_pos_matrix_dict = dict()
    band_freq_dict = defaultdict(lambda: defaultdict(list))
    band_tuning_index_dict = defaultdict(lambda: defaultdict(list))
    with h5py.File(replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, ['shared_context', 'decoded_pos_matrix'])
        for pop_name in group:
            decoded_pos_matrix_dict[pop_name] = group[pop_name][:]
        exported_data_group_key = 'simple_network_exported_data'
        for trial_key in (key for key in f if exported_data_group_key in f[key]):
            group = get_h5py_group(f, [trial_key, exported_data_group_key, 'filter_results'])
            subgroup = group['centroid_freq']
            for band in subgroup:
                for pop_name in subgroup[band].attrs:
                    band_freq_dict[band][pop_name].append(subgroup[band].attrs[pop_name])
            subgroup = group['freq_tuning_index']
            for band in subgroup:
                for pop_name in subgroup[band].attrs:
                    band_tuning_index_dict[band][pop_name].append(subgroup[band].attrs[pop_name])

    all_decoded_pos_dict = dict()
    decoded_pos_diff_var_dict = defaultdict(list)
    decoded_path_len_dict = defaultdict(list)
    decoded_distance_dict = defaultdict(list)

    for pop_name in decoded_pos_matrix_dict:
        this_decoded_pos_matrix = (decoded_pos_matrix_dict[pop_name] - track_start)/ track_length
        clean_indexes = ~np.isnan(this_decoded_pos_matrix)
        all_decoded_pos_dict[pop_name] = this_decoded_pos_matrix[clean_indexes]
        this_decoded_pos_diff = np.diff(this_decoded_pos_matrix, axis=1)
        for trial in range(this_decoded_pos_diff.shape[0]):
            this_trial_diff = this_decoded_pos_diff[trial, :]
            clean_indexes = ~np.isnan(this_trial_diff)
            if len(clean_indexes) > 0:
                this_trial_diff = this_trial_diff[clean_indexes]
                this_trial_diff[np.where(this_trial_diff < -0.5)] += 1.
                this_trial_diff[np.where(this_trial_diff > 0.5)] -= 1.
                this_path_len = np.sum(np.abs(this_trial_diff))
                decoded_path_len_dict[pop_name].append(this_path_len)
                this_distance = np.sum(this_trial_diff)
                decoded_distance_dict[pop_name].append(this_distance)
                if len(clean_indexes) > 1:
                    this_trial_diff_var = np.var(this_trial_diff)
                    decoded_pos_diff_var_dict[pop_name].append(this_trial_diff_var)

    if plot:
        ordered_pop_names = ['FF', 'E', 'I']
        for pop_name in ordered_pop_names:
            if pop_name not in decoded_pos_matrix_dict:
                ordered_pop_names.remove(pop_name)
        for pop_name in decoded_pos_matrix_dict:
            if pop_name not in ordered_pop_names:
                ordered_pop_names.append(pop_name)
        fig, axes = plt.subplots(3, 2, figsize=(8.5, 10.5), constrained_layout=True)

        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(all_decoded_pos_dict[pop_name], bins=np.linspace(0., 1., 21), density=True)
            bin_width = (edges[1] - edges[0])
            axes[1][0].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[1][0].set_xlim((0., 1.))
        axes[1][0].set_ylim((0., axes[1][0].get_ylim()[1]))
        axes[1][0].set_xlabel('Decoded position')
        axes[1][0].set_ylabel('Probability')
        axes[1][0].legend(loc='best', frameon=False, framealpha=0.5)
        axes[1][0].set_title('Decoded positions')

        max_variance = np.max(list(decoded_pos_diff_var_dict.values()))
        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(decoded_pos_diff_var_dict[pop_name], bins=np.linspace(0., max_variance, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            axes[0][1].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[0][1].set_xlim((0., max_variance))
        axes[0][1].set_ylim((0., axes[0][1].get_ylim()[1]))
        axes[0][1].set_xlabel('Variance')
        axes[0][1].set_ylabel('Probability')
        axes[0][1].legend(loc='best', frameon=False, framealpha=0.5)
        axes[0][1].set_title('Variance of steps within decoded trajectory')

        max_path_len = np.max(list(decoded_path_len_dict.values()))
        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(decoded_path_len_dict[pop_name], bins=np.linspace(0., max_path_len, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            axes[0][0].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[0][0].set_xlim((0., max_path_len))
        axes[0][0].set_ylim((0., axes[0][0].get_ylim()[1]))
        axes[0][0].set_xlabel('Path length')
        axes[0][0].set_ylabel('Probability')
        axes[0][0].legend(loc='best', frameon=False, framealpha=0.5)
        axes[0][0].set_title('Path length of decoded trajectory')

        max_distance = np.max(np.abs(list(decoded_distance_dict.values())))
        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(decoded_distance_dict[pop_name],
                                       bins=np.linspace(-max_distance, max_distance, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            axes[1][1].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[1][1].set_xlim((-max_distance, max_distance))
        axes[1][1].set_ylim((0., axes[1][1].get_ylim()[1]))
        axes[1][1].set_xlabel('Distance')
        axes[1][1].set_ylabel('Probability')
        axes[1][1].legend(loc='best', frameon=False, framealpha=0.5)
        axes[1][1].set_title('Distance traveled by decoded trajectory')

        min_freq = np.min(list(band_freq_dict['Ripple'].values()))
        max_freq = np.max(list(band_freq_dict['Ripple'].values()))
        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(band_freq_dict['Ripple'][pop_name], bins=np.linspace(min_freq, max_freq, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            axes[2][0].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[2][0].set_xlim((min_freq, max_freq))
        axes[2][0].set_ylim((0., axes[2][0].get_ylim()[1]))
        axes[2][0].set_xlabel('Frequency (Hz)')
        axes[2][0].set_ylabel('Probability')
        axes[2][0].legend(loc='best', frameon=False, framealpha=0.5)
        axes[2][0].set_title('Oscillation frequency')

        min_tuning_index = np.min(list(band_tuning_index_dict['Ripple'].values()))
        max_tuning_index = np.max(list(band_tuning_index_dict['Ripple'].values()))
        for pop_name in ordered_pop_names:
            hist, edges = np.histogram(band_tuning_index_dict['Ripple'][pop_name],
                                       bins=np.linspace(min_tuning_index, max_tuning_index, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            axes[2][1].plot(edges[1:] - bin_width / 2., hist * bin_width, label=pop_name)
        axes[2][1].set_xlim((min_tuning_index, max_tuning_index))
        axes[2][1].set_ylim((0., axes[2][1].get_ylim()[1]))
        axes[2][1].set_xlabel('Frequency tuning index')
        axes[2][1].set_ylabel('Probability')
        axes[2][1].legend(loc='best', frameon=False, framealpha=0.5)
        axes[2][1].set_title('Oscillation frequency tuning index')

        clean_axes(axes)
        fig.set_constrained_layout_pads(hspace=0.15, wspace=0.15)
        fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
