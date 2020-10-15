from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--run-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--replay-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--export-data-key", type=int, default=0)
@click.option("--plot-n-trials", type=int, default=10)
@click.option("--run-window-dur", type=float, default=20.)
@click.option("--replay-window-dur", type=float, default=20.)
@click.option("--step-dur", type=float, default=20.)
@click.option("--output-dir", type=str, default='data')
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--export", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, run_data_file_path, replay_data_file_path, export_data_key, plot_n_trials, run_window_dur,
         replay_window_dur, step_dur, output_dir, interactive, verbose, export, plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param run_data_file_path: str (path)
    :param replay_data_file_path: str (path)
    :param export_data_key: str
    :param plot_n_trials: int
    :param run_window_dur: float
    :param replay_window_dur: float
    :param step_dur: float
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

    config_parallel_interface(__file__, disp=context.disp, interface=context.interface, output_dir=output_dir,
                              verbose=verbose, plot=plot, debug=debug, export_data_key=export_data_key, export=export,
                              **kwargs)

    context.interface.update_worker_contexts(run_duration=context.run_duration, run_window_dur=context.run_window_dur,
                                             replay_duration=context.replay_duration,
                                             replay_window_dur=context.replay_window_dur)

    start_time = time.time()

    if not context.run_data_is_processed:
        context.sorted_gid_dict, context.run_firing_rate_matrix_dict = \
            process_run_data(context.run_data_file_path, context.run_trial_keys, context.export_data_key,
                             context.run_window_dur, export=context.export, disp=context.disp)
    else:
        context.sorted_gid_dict, context.run_firing_rate_matrix_dict = \
            load_processed_run_data(context.run_data_file_path, context.export_data_key)
    context.interface.update_worker_contexts(sorted_gid_dict=context.sorted_gid_dict,
                                             run_firing_rate_matrix_dict=context.run_firing_rate_matrix_dict)

    if context.disp:
        print('decode_simple_network_replay: processing run data for %i trials took %.1f s' %
              (len(context.run_trial_keys), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    if not context.replay_data_is_processed:
        # return dicts to analyze
        decoded_pos_matrix_dict = \
            process_replay_data(context.replay_data_file_path, context.replay_trial_keys, context.export_data_key,
                                context.replay_window_dur, context.replay_duration, context.run_window_dur,
                                context.run_duration, export=context.export, disp=context.disp, plot=context.plot,
                                plot_trial_keys=context.plot_replay_trial_keys)
    else:
        decoded_pos_matrix_dict = load_processed_replay_data(context.replay_data_file_path, context.export_data_key)
        if plot and context.plot_n_trials > 0:
            discard = process_replay_data(context.replay_data_file_path, context.plot_replay_trial_keys,
                                          context.export_data_key, context.replay_window_dur, context.replay_duration,
                                          context.run_window_dur, context.run_duration, export=False, temp_export=False,
                                          disp=context.disp, plot=True, plot_trial_keys=context.plot_replay_trial_keys)

    if context.disp:
        print('decode_simple_network_replay: processing replay data for %i trials took %.1f s' %
              (len(context.replay_trial_keys), time.time() - current_time))
        sys.stdout.flush()

    if plot:
        analyze_decoded_trajectory_data(decoded_pos_matrix_dict, context.replay_window_dur, context.run_duration,
                                        plot=True)

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

    run_group_key = 'simple_network_exported_run_data'
    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(context.run_data_file_path, 'r') as f:
        group = get_h5py_group(f, [context.export_data_key, run_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        run_duration = get_h5py_attr(subgroup.attrs, 'duration')
        run_binned_t_edges = np.arange(0., run_duration + context.run_window_dur / 2., context.run_window_dur)
        run_binned_t = run_binned_t_edges[:-1] + context.run_window_dur / 2.
        run_trial_keys = [key for key in group if key != shared_context_key]

        group = get_h5py_group(f, [context.export_data_key])
        if processed_group_key in group and shared_context_key in group[processed_group_key] and \
                'trial_averaged_firing_rate_matrix' in group[processed_group_key][shared_context_key]:
            run_data_is_processed = True
        else:
            run_data_is_processed = False

    if not os.path.isfile(context.replay_data_file_path):
        raise IOError('decode_simple_network_replay: invalid replay_data_file_path: %s' %
                      context.replay_data_file_path)
    replay_group_key = 'simple_network_exported_replay_data'
    with h5py.File(context.replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [context.export_data_key, replay_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        replay_duration = get_h5py_attr(subgroup.attrs, 'duration')
        replay_binned_t_edges = np.arange(0., replay_duration + context.replay_window_dur / 2.,
                                          context.replay_window_dur)
        replay_binned_t = replay_binned_t_edges[:-1] + context.replay_window_dur / 2.
        replay_trial_keys = [key for key in group if key != shared_context_key]
        context.plot_n_trials = min(int(context.plot_n_trials), len(replay_trial_keys))
        plot_replay_trial_keys = \
            list(np.random.choice(replay_trial_keys, context.plot_n_trials, replace=False))

        group = get_h5py_group(f, [context.export_data_key])
        if processed_group_key in group and shared_context_key in group[processed_group_key] and \
                'decoded_pos_matrix' in group[processed_group_key][shared_context_key]:
            replay_data_is_processed = True
        else:
            replay_data_is_processed = False

    context.update(locals())


def process_run_single_trial(run_data_file_path, trial_key, export_data_key, bin_dur, disp=True):
    """

    :param run_data_file_path: str
    :param trial_key: str
    :param export_data_key: str
    :param bin_dur: float
    :param disp: bool
    """
    start_time = time.time()
    full_spike_times_dict = dict()
    tuning_peak_locs = dict()

    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'
    with h5py.File(run_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, group_key])
        subgroup = group[shared_context_key]
        if 'tuning_peak_locs' in subgroup and len(subgroup['tuning_peak_locs']) > 0:
            data_group = subgroup['tuning_peak_locs']
            for pop_name in data_group:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(data_group[pop_name]['target_gids'], data_group[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        duration = get_h5py_attr(subgroup.attrs, 'duration')
        run_binned_t_edges = np.arange(0., duration + bin_dur / 2., bin_dur)
        run_binned_t = run_binned_t_edges[:-1] + bin_dur / 2.
        subgroup = get_h5py_group(group, [trial_key])
        data_group = subgroup['full_spike_times']
        for pop_name in data_group:
            full_spike_times_dict[pop_name] = dict()
            for gid_key in data_group[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]

    binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, run_binned_t_edges)

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

    binned_spike_count_matrix_dict = dict()
    for pop_name in binned_spike_count_dict:
        binned_spike_count_matrix_dict[pop_name] = np.empty((len(binned_spike_count_dict[pop_name]), len(run_binned_t)))
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            binned_spike_count_matrix_dict[pop_name][i, :] = binned_spike_count_dict[pop_name][gid]

    if disp:
        print('process_run_single_trial: pid: %i took %.1f s to process binned spike count data for trial: %s' %
              (os.getpid(), time.time() - start_time, trial_key))
        sys.stdout.flush()

    group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(context.temp_output_path, 'a') as f:
        group = get_h5py_group(f, [export_data_key, group_key], create=True)
        if shared_context_key not in group:
            subgroup = get_h5py_group(group, [shared_context_key], create=True)
            data_group = subgroup.create_group('sorted_gids')
            for pop_name in sorted_gid_dict:
                data_group.create_dataset(pop_name, data=sorted_gid_dict[pop_name], compression='gzip')
        subgroup = get_h5py_group(group, [trial_key], create=True)
        data_group = subgroup.create_group('binned_spike_count_matrix')
        for pop_name in binned_spike_count_matrix_dict:
            data_group.create_dataset(pop_name, data=binned_spike_count_matrix_dict[pop_name], compression='gzip')

    if disp:
        print('process_run_single_trial: pid: %i exported data for trial: %s to temp_output_path: %s' %
              (os.getpid(), trial_key, context.temp_output_path))
        sys.stdout.flush()


def process_run_data(run_data_file_path, run_trial_keys, export_data_key, bin_dur, export=True, disp=True):
    """

    :param run_data_file_path: str (path)
    :param run_trial_keys: list of str
    :param export_data_key: str
    :param bin_dur: float
    :param export: bool
    :param disp: bool
    """
    start_time = time.time()
    num_trials = len(run_trial_keys)
    sequences = [[run_data_file_path] * num_trials, run_trial_keys, [export_data_key] * num_trials,
                 [bin_dur] * num_trials, [disp] * num_trials]
    context.interface.map(process_run_single_trial, *sequences)
    temp_output_path_list = [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
                             if os.path.isfile(temp_output_path)]

    binned_spike_count_matrix_dict_trial_list = []
    first = True
    group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    for temp_output_path in temp_output_path_list:
        with h5py.File(temp_output_path, 'r') as f:
            group = get_h5py_group(f, [export_data_key, group_key])
            if first:
                subgroup = get_h5py_group(group, [shared_context_key])
                sorted_gid_dict = dict()
                data_group = subgroup['sorted_gids']
                for pop_name in data_group:
                    sorted_gid_dict[pop_name] = data_group[pop_name][:]
                first = False
            for trial_key in (key for key in group if key != shared_context_key):
                subgroup = get_h5py_group(group, [trial_key])
                data_group = subgroup['binned_spike_count_matrix']
                this_binned_spike_count_matrix_dict = dict()
                for pop_name in data_group:
                    this_binned_spike_count_matrix_dict[pop_name] = data_group[pop_name][:]
                binned_spike_count_matrix_dict_trial_list.append(this_binned_spike_count_matrix_dict)

    trial_averaged_binned_spike_count_matrix_dict = dict()
    for pop_name in binned_spike_count_matrix_dict_trial_list[0]:
        trial_averaged_binned_spike_count_matrix_dict[pop_name] = \
            np.mean([this_binned_spike_count_matrix_dict[pop_name]
                     for this_binned_spike_count_matrix_dict in binned_spike_count_matrix_dict_trial_list], axis=0)

    trial_averaged_run_firing_rate_matrix_dict = \
        get_firing_rates_from_binned_spike_count_matrix_dict(trial_averaged_binned_spike_count_matrix_dict,
                                                             bin_dur=bin_dur, smooth=150., wrap=True)
    if export:
        group_key = 'simple_network_processed_data'
        shared_context_key = 'shared_context'
        with h5py.File(run_data_file_path, 'a') as f:
            group = get_h5py_group(f, [export_data_key, group_key, shared_context_key], create=True)
            subgroup = group.create_group('sorted_gids')
            for pop_name in sorted_gid_dict:
                subgroup.create_dataset(pop_name, data=sorted_gid_dict[pop_name], compression='gzip')
            subgroup = group.create_group('trial_averaged_firing_rate_matrix')
            for pop_name in trial_averaged_run_firing_rate_matrix_dict:
                subgroup.create_dataset(pop_name, data=trial_averaged_run_firing_rate_matrix_dict[pop_name],
                                        compression='gzip')

        if disp:
            print('process_run_data: pid: %i; exporting to run_data_file_path: %s '
                  'took %.1f s' % (os.getpid(), run_data_file_path, time.time() - start_time))
            sys.stdout.flush()

    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)

    return sorted_gid_dict, trial_averaged_run_firing_rate_matrix_dict


def load_processed_run_data_helper(run_data_file_path, export_data_key):
    """

    :param run_data_file_path: str (path)
    :param export_data_key: str
    """
    context.sorted_gid_dict, context.run_firing_rate_matrix_dict = \
        load_processed_run_data(run_data_file_path, export_data_key)


def load_processed_run_data(run_data_file_path, export_data_key):
    """

    :param run_data_file_path: str (path)
    :param export_data_key: str
    :return: tuple of dict of array
    """
    sorted_gid_dict = dict()
    run_firing_rate_matrix_dict = dict()
    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(run_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, processed_group_key, shared_context_key])
        subgroup = group['sorted_gids']
        for pop_name in subgroup:
            sorted_gid_dict[pop_name] = subgroup[pop_name][:]
        subgroup = group['trial_averaged_firing_rate_matrix']
        for pop_name in subgroup:
            run_firing_rate_matrix_dict[pop_name] = subgroup[pop_name][:,:]

    return sorted_gid_dict, run_firing_rate_matrix_dict


def process_replay_single_trial_helper(replay_data_file_path, trial_key, export_data_key, replay_bin_dur,
                                       replay_duration, run_bin_dur, run_duration, export=False, disp=True, plot=False):
    """

    :param replay_data_file_path: str (path)
    :param trial_key: str
    :param export_data_key: str
    :param replay_bin_dur: float (ms)
    :param replay_duration: float (ms)
    :param run_bin_dur: float (ms)
    :param run_duration: float (ms)
    :param export: bool
    :param disp: bool
    :param plot: bool
    """
    replay_full_spike_times_dict = dict()
    replay_group_key = 'simple_network_exported_replay_data'
    with h5py.File(replay_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, replay_group_key])
        subgroup = get_h5py_group(group, [trial_key, 'full_spike_times'])
        for pop_name in subgroup:
            if pop_name not in replay_full_spike_times_dict:
                replay_full_spike_times_dict[pop_name] = dict()
            for gid_key in subgroup[pop_name]:
                replay_full_spike_times_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]

    decoded_pos_dict = \
        process_replay_single_trial(replay_full_spike_times_dict, trial_key, replay_bin_dur, replay_duration,
                                    run_bin_dur, run_duration, context.sorted_gid_dict,
                                    context.run_firing_rate_matrix_dict, disp=disp, plot=plot)

    if export:
        group_key = 'simple_network_processed_data'
        with h5py.File(context.temp_output_path, 'a') as f:
            group = get_h5py_group(f, [export_data_key, group_key, trial_key], create=True)
            subgroup = group.create_group('decoded_position')
            for pop_name in decoded_pos_dict:
                subgroup.create_dataset(pop_name, data=decoded_pos_dict[pop_name], compression='gzip')

        if disp:
            print('process_replay_single_trial_helper: pid: %i exported data for trial: %s to temp_output_path: %s' %
                  (os.getpid(), trial_key, context.temp_output_path))
            sys.stdout.flush()


def process_replay_single_trial(replay_spike_times_dict, trial_key, replay_bin_dur, replay_duration, run_bin_dur,
                                run_duration, sorted_gid_dict, run_firing_rate_matrix_dict, disp=False, plot=False):
    """

    :param replay_spike_times_dict: dict {pop_name: {gid: array}}
    :param trial_key: str
    :param replay_bin_dur: float (ms)
    :param replay_duration: float (ms)
    :param run_bin_dur: float (ms)
    :param run_duration: float (ms)
    :param sorted_gid_dict: dict of array of int
    :param run_firing_rate_matrix_dict: dict of array of float
    :param disp: bool
    :param plot: bool
    """
    start_time = time.time()
    replay_binned_t_edges = np.arange(0., replay_duration + replay_bin_dur / 2., replay_bin_dur)
    replay_binned_t = replay_binned_t_edges[:-1] + replay_bin_dur / 2.
    run_binned_t_edges = np.arange(0., run_duration + run_bin_dur / 2., run_bin_dur)
    run_binned_t = run_binned_t_edges[:-1] + run_bin_dur / 2.

    replay_binned_spike_count_dict = \
        get_binned_spike_count_dict(replay_spike_times_dict, replay_binned_t_edges)

    replay_binned_spike_count_matrix_dict = {}
    for pop_name in replay_binned_spike_count_dict:
        replay_binned_spike_count_matrix = np.empty((len(replay_binned_spike_count_dict[pop_name]),
                                                     len(replay_binned_t)))
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            replay_binned_spike_count_matrix[i, :] = replay_binned_spike_count_dict[pop_name][gid]
        replay_binned_spike_count_matrix_dict[pop_name] = replay_binned_spike_count_matrix

    p_pos_dict = decode_position_from_offline_replay(replay_binned_spike_count_matrix_dict, run_firing_rate_matrix_dict,
                                                     bin_dur=replay_bin_dur)

    decoded_pos_dict = dict()
    for pop_name in p_pos_dict:
        this_decoded_pos = np.empty_like(replay_binned_t)
        this_decoded_pos[:] = np.nan
        p_pos = p_pos_dict[pop_name]
        for pos_bin in range(p_pos.shape[1]):
            if np.any(~np.isnan(p_pos[:, pos_bin])):
                index = np.nanargmax(p_pos[:, pos_bin])
                val = p_pos[index, pos_bin]
                if len(np.where(p_pos[:, pos_bin] == val)[0]) == 1:
                    this_decoded_pos[pos_bin] = run_binned_t[index]
        decoded_pos_dict[pop_name] = this_decoded_pos

    if disp:
        print('process_replay_single_trial: pid: %i took %.1f s to decode position for trial: %s' %
              (os.getpid(), time.time() - start_time, trial_key))
        sys.stdout.flush()

    if plot:
        ordered_pop_names = ['FF', 'E', 'I']
        for pop_name in ordered_pop_names:
            if pop_name not in p_pos_dict:
                ordered_pop_names.remove(pop_name)
        for pop_name in p_pos_dict:
            if pop_name not in ordered_pop_names:
                ordered_pop_names.append(pop_name)
        fig, axes = plt.subplots(2, len(ordered_pop_names), figsize=(3.8 * len(ordered_pop_names) + 0.5, 7.5))
        decoded_x_mesh, decoded_y_mesh = \
            np.meshgrid(replay_binned_t_edges, run_binned_t_edges)
        this_cmap = plt.get_cmap()
        this_cmap.set_bad(this_cmap(0.))
        for col, pop_name in enumerate(ordered_pop_names):
            p_pos = p_pos_dict[pop_name]
            axes[1][col].pcolormesh(decoded_x_mesh, decoded_y_mesh, p_pos, vmin=0.)
            axes[1][col].set_xlabel('Time (ms)')
            axes[1][col].set_ylim((run_binned_t_edges[-1], run_binned_t_edges[0]))
            axes[1][col].set_xlim((replay_binned_t_edges[0], replay_binned_t_edges[-1]))
            axes[1][col].set_ylabel('Decoded position')
            axes[1][col].set_title('Population: %s' % pop_name)

            for i, gid in enumerate(sorted_gid_dict[pop_name]):
                this_spike_times = replay_spike_times_dict[pop_name][gid]
                axes[0][col].scatter(this_spike_times, np.ones_like(this_spike_times) * i + 0.5, c='k', s=1.)
            axes[0][col].set_xlabel('Time (ms)')
            axes[0][col].set_ylim((len(sorted_gid_dict[pop_name]), 0))
            axes[0][col].set_xlim((replay_binned_t_edges[0], replay_binned_t_edges[-1]))
            axes[0][col].set_ylabel('Sorted cells')
            axes[0][col].set_title('Population: %s' % pop_name)
        fig.suptitle('Trial # %s' % trial_key, y=0.99)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9)

    return decoded_pos_dict


def process_replay_data(replay_data_file_path, replay_trial_keys, export_data_key, replay_bin_dur, replay_duration,
                        run_bin_dur, run_duration, export=False, temp_export=True, disp=True, plot=False,
                        plot_trial_keys=None):
    """

    :param replay_data_file_path: str (path)
    :param replay_trial_keys: list of str
    :param export_data_key: str
    :param replay_bin_dur: float (ms)
    :param replay_duration: float (ms)
    :param run_bin_dur: float (ms)
    :param run_duration: float (ms)
    :param export: bool
    :param temp_export: bool
    :param disp: bool
    :param plot: bool
    :param plot_trial_keys: list of str
    :return: dict: {pop_name: 2D array}
    """
    num_trials = len(replay_trial_keys)
    if not plot or plot_trial_keys is None or len(plot_trial_keys) == 0:
        plot_list = [False] * num_trials
    else:
        plot_list = []
        for trial_key in replay_trial_keys:
            if trial_key in plot_trial_keys:
                plot_list.append(True)
            else:
                plot_list.append(False)
    sequences = [[replay_data_file_path] * num_trials, replay_trial_keys, [export_data_key] * num_trials,
                 [replay_bin_dur] * num_trials, [replay_duration] * num_trials, [run_bin_dur] * num_trials,
                 [run_duration] * num_trials, [temp_export] * num_trials, [disp] * num_trials, plot_list]
    context.interface.map(process_replay_single_trial_helper, *sequences)

    decoded_pos_array_list_dict = dict()
    decoded_pos_matrix_dict = dict()

    if temp_export:
        start_time = time.time()
        temp_output_path_list = \
            [temp_output_path for temp_output_path in context.interface.get('context.temp_output_path')
             if os.path.isfile(temp_output_path)]
        trial_key_list = []

        group_key = 'simple_network_processed_data'
        shared_context_key = 'shared_context'
        for temp_output_path in temp_output_path_list:
            with h5py.File(temp_output_path, 'r') as f:
                group = get_h5py_group(f, [export_data_key, group_key])
                for trial_key in (key for key in group if key != shared_context_key):
                    trial_key_list.append(trial_key)
                    subgroup = get_h5py_group(group, [trial_key])
                    data_group = subgroup['decoded_position']
                    for pop_name in data_group:
                        this_decoded_position = data_group[pop_name][:]
                        if pop_name not in decoded_pos_array_list_dict:
                            decoded_pos_array_list_dict[pop_name] = []
                        decoded_pos_array_list_dict[pop_name].append(this_decoded_position)

        sorted_trial_indexes = np.argsort(np.array(trial_key_list, dtype=int))
        for pop_name in decoded_pos_array_list_dict:
            decoded_pos_matrix_dict[pop_name] = \
                np.asarray(decoded_pos_array_list_dict[pop_name])[sorted_trial_indexes,:]

        if export:
            with h5py.File(replay_data_file_path, 'a') as f:
                group = get_h5py_group(f, [export_data_key, group_key, shared_context_key], create=True)
                subgroup = group.create_group('decoded_pos_matrix')
                for pop_name in decoded_pos_matrix_dict:
                    subgroup.create_dataset(pop_name, data=decoded_pos_matrix_dict[pop_name], compression='gzip')
            if disp:
                print('process_replay_data: pid: %i; exporting to replay_data_file_path: %s took %.1f s' %
                      (os.getpid(), replay_data_file_path, time.time() - start_time))

        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)

    return decoded_pos_matrix_dict


def load_processed_replay_data(replay_data_file_path, export_data_key):
    """

    :param replay_data_file_path: str (path)
    :param export_data_key: str
    :return: dict: {pop_name: 2D array}
    """
    group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    decoded_pos_matrix_dict = dict()
    slope_array_dict = dict()
    p_val_array_dict = dict()
    with h5py.File(replay_data_file_path, 'a') as f:
        group = get_h5py_group(f, [export_data_key, group_key, shared_context_key])
        subgroup = get_h5py_group(group, ['decoded_pos_matrix'])
        for pop_name in subgroup:
            decoded_pos_matrix_dict[pop_name] = subgroup[pop_name][:,:]

    return decoded_pos_matrix_dict


def analyze_decoded_trajectory_data(decoded_pos_matrix_dict, bin_dur, run_duration, plot=True):
    """

    :param decoded_pos_matrix_dict: dict or list of dict: {pop_name: 2d array of float}
    :param bin_dur: float
    :param run_duration: float
    :param plot: bool
    :return:
    """
    if not isinstance(decoded_pos_matrix_dict, list):
        decoded_pos_matrix_dict_instances_list = [decoded_pos_matrix_dict]
    else:
        decoded_pos_matrix_dict_instances_list = decoded_pos_matrix_dict

    all_decoded_pos_instances_list_dict = defaultdict(list)
    decoded_velocity_var_instances_list_dict = defaultdict(list)
    decoded_path_len_instances_list_dict = defaultdict(list)
    decoded_velocity_mean_instances_list_dict = defaultdict(list)

    for decoded_pos_matrix_dict in decoded_pos_matrix_dict_instances_list:
        all_decoded_pos_dict = dict()
        decoded_path_len_dict = defaultdict(list)
        decoded_velocity_var_dict = defaultdict(list)
        decoded_velocity_mean_dict = defaultdict(list)
        for pop_name in decoded_pos_matrix_dict:
            this_decoded_pos_matrix = decoded_pos_matrix_dict[pop_name][:, :] / run_duration
            clean_indexes = ~np.isnan(this_decoded_pos_matrix)
            all_decoded_pos_dict[pop_name] = this_decoded_pos_matrix[clean_indexes]
            for trial in range(this_decoded_pos_matrix.shape[0]):
                this_trial_pos = this_decoded_pos_matrix[trial, :]
                clean_indexes = ~np.isnan(this_trial_pos)
                if len(clean_indexes) > 0:
                    this_trial_diff = np.diff(this_trial_pos[clean_indexes])
                    this_trial_diff[np.where(this_trial_diff < -0.5)] += 1.
                    this_trial_diff[np.where(this_trial_diff > 0.5)] -= 1.
                    this_path_len = np.sum(np.abs(this_trial_diff))
                    decoded_path_len_dict[pop_name].append(this_path_len)
                    this_trial_velocity = this_trial_diff / (bin_dur / 1000.)
                    this_trial_velocity_mean = np.mean(this_trial_velocity)
                    decoded_velocity_mean_dict[pop_name].append(this_trial_velocity_mean)
                    if len(clean_indexes) > 1:
                        this_trial_velocity_var = np.var(this_trial_velocity)
                        decoded_velocity_var_dict[pop_name].append(this_trial_velocity_var)
            all_decoded_pos_instances_list_dict[pop_name].append(all_decoded_pos_dict[pop_name])
            decoded_path_len_instances_list_dict[pop_name].append(decoded_path_len_dict[pop_name])
            decoded_velocity_var_instances_list_dict[pop_name].append(decoded_velocity_var_dict[pop_name])
            decoded_velocity_mean_instances_list_dict[pop_name].append(decoded_velocity_mean_dict[pop_name])

    ordered_pop_names = ['FF', 'E', 'I']
    for pop_name in ordered_pop_names:
        if pop_name not in all_decoded_pos_instances_list_dict:
            ordered_pop_names.remove(pop_name)
    for pop_name in all_decoded_pos_instances_list_dict:
        if pop_name not in ordered_pop_names:
            ordered_pop_names.append(pop_name)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), constrained_layout=True)

    max_vel_var = np.max(list(decoded_velocity_var_instances_list_dict.values()))
    max_path_len = np.max(list(decoded_path_len_instances_list_dict.values()))
    max_vel_mean = np.max(list(decoded_velocity_mean_instances_list_dict.values()))
    min_vel_mean = np.min(list(decoded_velocity_mean_instances_list_dict.values()))

    num_instances = len(decoded_pos_matrix_dict_instances_list)
    for pop_name in ordered_pop_names:
        hist_list = []
        for all_decoded_pos in all_decoded_pos_instances_list_dict[pop_name]:
            hist, edges = np.histogram(all_decoded_pos, bins=np.linspace(0., 1., 21), density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            axes[1][0].plot(edges[1:] - bin_width / 2., hist_list[0], label=pop_name)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            axes[1][0].plot(edges[1:] - bin_width / 2., mean_hist, label=pop_name)
            axes[1][0].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0)

        hist_list = []
        for decoded_velocity_var in decoded_velocity_var_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_velocity_var, bins=np.linspace(0., max_vel_var, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            axes[0][1].plot(edges[1:] - bin_width / 2., hist_list[0], label=pop_name)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            axes[0][1].plot(edges[1:] - bin_width / 2., mean_hist, label=pop_name)
            axes[0][1].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0)

        hist_list = []
        for decoded_path_len in decoded_path_len_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_path_len, bins=np.linspace(0., max_path_len, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            axes[0][0].plot(edges[1:] - bin_width / 2., hist_list[0], label=pop_name)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            axes[0][0].plot(edges[1:] - bin_width / 2., mean_hist, label=pop_name)
            axes[0][0].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0)

        hist_list = []
        for decoded_velocity_mean in decoded_velocity_mean_instances_list_dict[pop_name]:
            hist, edges = np.histogram(decoded_velocity_mean, bins=np.linspace(min_vel_mean, max_vel_mean, 21),
                                       density=True)
            bin_width = (edges[1] - edges[0])
            hist *= bin_width
            hist_list.append(hist)
        if num_instances == 1:
            axes[1][1].plot(edges[1:] - bin_width / 2., hist_list[0], label=pop_name)
        else:
            mean_hist = np.mean(hist_list, axis=0)
            mean_sem = np.std(hist_list, axis=0) / np.sqrt(num_instances)
            axes[1][1].plot(edges[1:] - bin_width / 2., mean_hist, label=pop_name)
            axes[1][1].fill_between(edges[1:] - bin_width / 2., mean_hist + mean_sem, mean_hist - mean_sem,
                                    alpha=0.25, linewidth=0)

    axes[0][0].set_xlim((0., max_path_len))
    axes[0][0].set_ylim((0., axes[0][0].get_ylim()[1]))
    axes[0][0].set_xlabel('Normalized path length')
    axes[0][0].set_ylabel('Probability')
    axes[0][0].legend(loc='best', frameon=False, framealpha=0.5)
    axes[0][0].set_title('Path length of decoded trajectories')

    axes[0][1].set_xlim((0., max_vel_var))
    axes[0][1].set_ylim((0., axes[0][1].get_ylim()[1]))
    axes[0][1].set_xlabel('Variance (/s^2)')
    axes[0][1].set_ylabel('Probability')
    axes[0][1].legend(loc='best', frameon=False, framealpha=0.5)
    axes[0][1].set_title('Variance of velocity of decoded trajectories')

    axes[1][0].set_xlim((0., 1.))
    axes[1][0].set_ylim((0., axes[1][0].get_ylim()[1]))
    axes[1][0].set_xlabel('Normalized position')
    axes[1][0].set_ylabel('Probability')
    axes[1][0].legend(loc='best', frameon=False, framealpha=0.5)
    axes[1][0].set_title('Decoded positions')

    axes[1][1].set_xlim((min_vel_mean, max_vel_mean))
    axes[1][1].set_ylim((0., axes[1][1].get_ylim()[1]))
    axes[1][1].set_xlabel('Trajectory velocity (/s)')
    axes[1][1].set_ylabel('Probability')
    axes[1][1].legend(loc='best', frameon=False, framealpha=0.5)
    axes[1][1].set_title('Mean velocity of decoded trajectories')

    clean_axes(axes)
    fig.set_constrained_layout_pads(hspace=0.15, wspace=0.15)
    fig.show()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
