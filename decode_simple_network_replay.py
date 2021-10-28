from nested.parallel import *
from nested.optimize_utils import *
from simple_network_analysis_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--template-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--decode-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--template-group-key", type=str, default='simple_network_exported_run_data')
@click.option("--decode-group-key", type=str, default='simple_network_exported_replay_data')
@click.option("--export-data-key", type=int, default=0)
@click.option("--plot-n-trials", type=int, default=10)
@click.option("--plot-trials", '-pt', type=int, multiple=True)
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False,
              default=None)
@click.option("--template-window-dur", type=float, default=20.)
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--step-dur", type=float, default=20.)
@click.option("--output-dir", type=str, default='data')
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--export", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--compressed-plot-format", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, template_data_file_path, decode_data_file_path, template_group_key, decode_group_key, export_data_key,
         plot_n_trials, plot_trials, config_file_path, template_window_dur, decode_window_dur, step_dur, output_dir,
         interactive, verbose, export, plot, compressed_plot_format, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param template_data_file_path: str (path)
    :param decode_data_file_path: str (path)
    :param template_group_key: str
    :param decode_group_key: str
    :param export_data_key: str
    :param plot_n_trials: int
    :param plot_trials: list of int
    :param config_file_path: str (path); contains plot settings
    :param template_window_dur: float
    :param decode_window_dur: float
    :param step_dur: float
    :param output_dir: str (path to dir)
    :param interactive: bool
    :param verbose: int
    :param export: bool
    :param plot: bool
    :param compressed_plot_format: bool
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
                              template_group_key=template_group_key, decode_group_key=decode_group_key, **kwargs)

    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None

    start_time = time.time()

    if not context.template_data_is_processed:
        gid_order_dict, context.template_firing_rate_matrix_dict = \
            process_template_data(context.template_data_file_path, context.template_trial_keys,
                                  context.template_group_key, context.export_data_key, context.template_window_dur,
                                  export=context.export, disp=context.disp)
    else:
        gid_order_dict, context.template_firing_rate_matrix_dict = \
            load_processed_template_data(context.template_data_file_path, context.export_data_key)
    _, _, context.sorted_gid_dict = \
        analyze_selectivity_from_firing_rate_matrix_dict(context.template_firing_rate_matrix_dict,
                                                         gid_order_dict)
    context.template_firing_rate_matrix_dict = \
        sort_firing_rate_matrix_dict(context.template_firing_rate_matrix_dict, gid_order_dict,
                                     context.sorted_gid_dict)

    if context.plot:
        plot_firing_rate_heatmaps_from_matrix(context.template_firing_rate_matrix_dict,
                                              context.template_binned_t_edges, gids_sorted=True,
                                              pop_order=context.pop_order, label_dict=context.label_dict,
                                              normalize_t=False)

    context.interface.update_worker_contexts(sorted_gid_dict=context.sorted_gid_dict,
                                             template_firing_rate_matrix_dict=context.template_firing_rate_matrix_dict)

    if context.disp:
        print('decode_simple_network_replay: processing template data for %i trials took %.1f s' %
              (len(context.template_trial_keys), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    if not context.decode_data_is_processed:
        decoded_pos_matrix_dict = \
            decode_data(context.decode_data_file_path, context.decode_trial_keys, context.decode_group_key,
                        context.export_data_key, context.decode_window_dur, context.decode_duration,
                        context.template_window_dur, context.template_duration, export=context.export,
                        pop_order=context.pop_order, label_dict=context.label_dict, disp=context.disp,
                        plot=context.plot, plot_trial_keys=context.plot_decode_trial_keys,
                        compressed_plot_format=compressed_plot_format)
    else:
        decoded_pos_matrix_dict = load_decoded_data(context.decode_data_file_path, context.export_data_key)
        if plot and context.plot_n_trials > 0:
            discard = decode_data(context.decode_data_file_path, context.plot_decode_trial_keys,
                                  context.decode_group_key, context.export_data_key, context.decode_window_dur,
                                  context.decode_duration, context.template_window_dur, context.template_duration,
                                  export=False, temp_export=False, pop_order=context.pop_order,
                                  label_dict=context.label_dict, disp=context.disp, plot=True,
                                  plot_trial_keys=context.plot_decode_trial_keys,
                                  compressed_plot_format=compressed_plot_format)

    if context.disp:
        print('decode_simple_network_replay: decoding data for %i trials took %.1f s' %
              (len(context.decode_trial_keys), time.time() - current_time))
        sys.stdout.flush()

    if context.plot:
        plot_decoded_trajectory_replay_data(decoded_pos_matrix_dict, context.decode_window_dur,
                                            context.template_duration, pop_order=context.pop_order,
                                            label_dict=context.label_dict, color_dict=context.color_dict)
        context.interface.apply(plt.show)
        plt.show()

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


def config_controller():

    if not os.path.isfile(context.template_data_file_path):
        raise IOError('decode_simple_network_replay: invalid template_data_file_path: %s' %
                      context.template_data_file_path)

    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(context.template_data_file_path, 'r') as f:
        group = get_h5py_group(f, [context.export_data_key, context.template_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        template_duration = get_h5py_attr(subgroup.attrs, 'duration')
        template_binned_t_edges = np.arange(0., template_duration + context.template_window_dur / 2.,
                                            context.template_window_dur)
        template_binned_t = template_binned_t_edges[:-1] + context.template_window_dur / 2.
        template_trial_keys = [key for key in group if key != shared_context_key]

        group = get_h5py_group(f, [context.export_data_key])
        if processed_group_key in group and shared_context_key in group[processed_group_key] and \
                'trial_averaged_firing_rate_matrix' in group[processed_group_key][shared_context_key]:
            template_data_is_processed = True
        else:
            template_data_is_processed = False

    if not os.path.isfile(context.decode_data_file_path):
        raise IOError('decode_simple_network_replay: invalid decode_data_file_path: %s' %
                      context.decode_data_file_path)

    with h5py.File(context.decode_data_file_path, 'r') as f:
        group = get_h5py_group(f, [context.export_data_key, context.decode_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        decode_duration = get_h5py_attr(subgroup.attrs, 'duration')
        decode_binned_t_edges = np.arange(0., decode_duration + context.decode_window_dur / 2.,
                                            context.decode_window_dur)
        decode_binned_t = decode_binned_t_edges[:-1] + context.decode_window_dur / 2.
        decode_trial_keys = [key for key in group if key != shared_context_key]
        context.plot_n_trials = min(int(context.plot_n_trials), len(decode_trial_keys))
        plot_decode_trial_keys = []
        for plot_trial in context.plot_trials:
            plot_trial_key = str(plot_trial)
            if plot_trial_key not in decode_trial_keys:
                raise RuntimeError('decode_simple_network_replay: trial: %s not found in decode_data_file_path: %s' %
                                   (plot_trial_key, context.decode_data_file_path))
            plot_decode_trial_keys.append(plot_trial_key)
        plot_decode_trial_keys.extend(list(
            np.random.choice(decode_trial_keys, context.plot_n_trials - len(plot_decode_trial_keys), replace=False)))

        group = get_h5py_group(f, [context.export_data_key])
        if processed_group_key in group and shared_context_key in group[processed_group_key] and \
                'decoded_pos_matrix' in group[processed_group_key][shared_context_key]:
            decode_data_is_processed = True
        else:
            decode_data_is_processed = False

    context.update(locals())


def process_template_single_trial(template_data_file_path, trial_key, group_key, export_data_key, bin_dur, disp=True):
    """

    :param template_data_file_path: str
    :param trial_key: str
    :param group_key: str
    :param export_data_key: str
    :param bin_dur: float
    :param disp: bool
    """
    start_time = time.time()
    full_spike_times_dict = dict()
    tuning_peak_locs = dict()

    shared_context_key = 'shared_context'
    with h5py.File(template_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, group_key])
        subgroup = group[shared_context_key]
        if 'tuning_peak_locs' in subgroup and len(subgroup['tuning_peak_locs']) > 0:
            data_group = subgroup['tuning_peak_locs']
            for pop_name in data_group:
                tuning_peak_locs[pop_name] = dict()
                for target_gid, peak_loc in zip(data_group[pop_name]['target_gids'], data_group[pop_name]['peak_locs']):
                    tuning_peak_locs[pop_name][target_gid] = peak_loc
        duration = get_h5py_attr(subgroup.attrs, 'duration')
        template_binned_t_edges = np.arange(0., duration + bin_dur / 2., bin_dur)
        template_binned_t = template_binned_t_edges[:-1] + bin_dur / 2.
        subgroup = get_h5py_group(group, [trial_key])
        data_group = subgroup['full_spike_times']
        for pop_name in data_group:
            full_spike_times_dict[pop_name] = dict()
            for gid_key in data_group[pop_name]:
                full_spike_times_dict[pop_name][int(gid_key)] = data_group[pop_name][gid_key][:]

    binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, template_binned_t_edges)

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
        binned_spike_count_matrix_dict[pop_name] = np.empty((len(binned_spike_count_dict[pop_name]),
                                                             len(template_binned_t)))
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            binned_spike_count_matrix_dict[pop_name][i, :] = binned_spike_count_dict[pop_name][gid]

    if disp:
        print('process_template_single_trial: pid: %i took %.1f s to process binned spike count data for trial: %s' %
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
        print('process_template_single_trial: pid: %i exported data for trial: %s to temp_output_path: %s' %
              (os.getpid(), trial_key, context.temp_output_path))
        sys.stdout.flush()


def process_template_data(template_data_file_path, template_trial_keys, template_group_key, export_data_key, bin_dur,
                          export=True, disp=True):
    """

    :param template_data_file_path: str (path)
    :param template_trial_keys: list of str
    :param template_group_key: str
    :param export_data_key: str
    :param bin_dur: float
    :param export: bool
    :param disp: bool
    """
    start_time = time.time()
    num_trials = len(template_trial_keys)
    sequences = [[template_data_file_path] * num_trials, template_trial_keys, [template_group_key] * num_trials,
                 [export_data_key] * num_trials, [bin_dur] * num_trials, [disp] * num_trials]
    context.interface.map(process_template_single_trial, *sequences)
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

    trial_averaged_template_firing_rate_matrix_dict = \
        get_firing_rates_from_binned_spike_count_matrix_dict(trial_averaged_binned_spike_count_matrix_dict,
                                                             bin_dur=bin_dur, smooth=150., wrap=True)
    if export:
        group_key = 'simple_network_processed_data'
        shared_context_key = 'shared_context'
        with h5py.File(template_data_file_path, 'a') as f:
            group = get_h5py_group(f, [export_data_key, group_key, shared_context_key], create=True)
            subgroup = group.create_group('sorted_gids')
            for pop_name in sorted_gid_dict:
                subgroup.create_dataset(pop_name, data=sorted_gid_dict[pop_name], compression='gzip')
            subgroup = group.create_group('trial_averaged_firing_rate_matrix')
            for pop_name in trial_averaged_template_firing_rate_matrix_dict:
                subgroup.create_dataset(pop_name, data=trial_averaged_template_firing_rate_matrix_dict[pop_name],
                                        compression='gzip')

        if disp:
            print('process_template_data: pid: %i; exporting to template_data_file_path: %s '
                  'took %.1f s' % (os.getpid(), template_data_file_path, time.time() - start_time))
            sys.stdout.flush()

    for temp_output_path in temp_output_path_list:
        os.remove(temp_output_path)

    return sorted_gid_dict, trial_averaged_template_firing_rate_matrix_dict


def load_processed_template_data(template_data_file_path, export_data_key, plot=False):
    """

    :param template_data_file_path: str (path)
    :param export_data_key: str
    :param plot: bool
    :return: tuple of dict of array
    """
    sorted_gid_dict = dict()
    template_firing_rate_matrix_dict = dict()
    processed_group_key = 'simple_network_processed_data'
    shared_context_key = 'shared_context'
    with h5py.File(template_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, processed_group_key, shared_context_key])
        subgroup = group['sorted_gids']
        for pop_name in subgroup:
            sorted_gid_dict[pop_name] = subgroup[pop_name][:]
        subgroup = group['trial_averaged_firing_rate_matrix']
        for pop_name in subgroup:
            template_firing_rate_matrix_dict[pop_name] = subgroup[pop_name][:,:]

    return sorted_gid_dict, template_firing_rate_matrix_dict


def decode_single_trial_helper(decode_data_file_path, trial_key, decode_group_key, export_data_key, decode_bin_dur,
                               decode_duration, template_bin_dur, template_duration, export=False, pop_order=None,
                               label_dict=None, disp=True, plot=False, compressed_plot_format=False):
    """

    :param decode_data_file_path: str (path)
    :param trial_key: str
    :param decode_group_key: str
    :param export_data_key: str
    :param decode_bin_dur: float (ms)
    :param decode_duration: float (ms)
    :param template_bin_dur: float (ms)
    :param template_duration: float (ms)
    :param export: bool
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param disp: bool
    :param plot: bool
    :param compressed_plot_format: bool
    """
    decode_full_spike_times_dict = dict()
    with h5py.File(decode_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, decode_group_key])
        subgroup = get_h5py_group(group, [trial_key, 'full_spike_times'])
        for pop_name in subgroup:
            if pop_name not in decode_full_spike_times_dict:
                decode_full_spike_times_dict[pop_name] = dict()
            for gid_key in subgroup[pop_name]:
                decode_full_spike_times_dict[pop_name][int(gid_key)] = subgroup[pop_name][gid_key][:]

    decoded_pos_dict = \
        decode_single_trial(decode_full_spike_times_dict, trial_key, decode_bin_dur, decode_duration, template_bin_dur,
                            template_duration, context.sorted_gid_dict, context.template_firing_rate_matrix_dict,
                            pop_order=pop_order, label_dict=label_dict, disp=disp, plot=plot,
                            compressed_plot_format=compressed_plot_format)

    if export:
        group_key = 'simple_network_processed_data'
        with h5py.File(context.temp_output_path, 'a') as f:
            group = get_h5py_group(f, [export_data_key, group_key, trial_key], create=True)
            subgroup = group.create_group('decoded_position')
            for pop_name in decoded_pos_dict:
                subgroup.create_dataset(pop_name, data=decoded_pos_dict[pop_name], compression='gzip')

        if disp:
            print('decode_single_trial_helper: pid: %i exported data for trial: %s to temp_output_path: %s' %
                  (os.getpid(), trial_key, context.temp_output_path))
            sys.stdout.flush()


def decode_single_trial(decode_spike_times_dict, trial_key, decode_bin_dur, decode_duration, template_bin_dur,
                        template_duration, sorted_gid_dict, template_firing_rate_matrix_dict, pop_order=None,
                        label_dict=None, disp=False, plot=False, compressed_plot_format=False):
    """

    :param decode_spike_times_dict: dict {pop_name: {gid: array}}
    :param trial_key: str
    :param decode_bin_dur: float (ms)
    :param decode_duration: float (ms)
    :param template_bin_dur: float (ms)
    :param template_duration: float (ms)
    :param sorted_gid_dict: dict of array of int
    :param template_firing_rate_matrix_dict: dict of array of float
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param disp: bool
    :param plot: bool
    :param compressed_plot_format: bool
    """
    start_time = time.time()
    decode_binned_t_edges = np.arange(0., decode_duration + decode_bin_dur / 2., decode_bin_dur)
    decode_binned_t = decode_binned_t_edges[:-1] + decode_bin_dur / 2.
    template_binned_t_edges = np.arange(0., template_duration + template_bin_dur / 2., template_bin_dur)
    template_binned_t = template_binned_t_edges[:-1] + template_bin_dur / 2.

    decode_binned_spike_count_dict = \
        get_binned_spike_count_dict(decode_spike_times_dict, decode_binned_t_edges)

    decode_binned_spike_count_matrix_dict = {}
    for pop_name in decode_binned_spike_count_dict:
        decode_binned_spike_count_matrix = np.empty((len(decode_binned_spike_count_dict[pop_name]),
                                                     len(decode_binned_t)))
        for i, gid in enumerate(sorted_gid_dict[pop_name]):
            decode_binned_spike_count_matrix[i, :] = decode_binned_spike_count_dict[pop_name][gid]
        decode_binned_spike_count_matrix_dict[pop_name] = decode_binned_spike_count_matrix

    p_pos_dict = decode_position(decode_binned_spike_count_matrix_dict, template_firing_rate_matrix_dict,
                                 bin_dur=decode_bin_dur)

    decoded_pos_dict = dict()
    for pop_name in p_pos_dict:
        this_decoded_pos = np.empty_like(decode_binned_t)
        this_decoded_pos[:] = np.nan
        p_pos = p_pos_dict[pop_name]
        for pos_bin in range(p_pos.shape[1]):
            if np.any(~np.isnan(p_pos[:, pos_bin])):
                index = np.nanargmax(p_pos[:, pos_bin])
                val = p_pos[index, pos_bin]
                if len(np.where(p_pos[:, pos_bin] == val)[0]) == 1:
                    this_decoded_pos[pos_bin] = template_binned_t[index]
        decoded_pos_dict[pop_name] = this_decoded_pos

    if disp:
        print('decode_single_trial: pid: %i took %.1f s to decode position for trial: %s' %
              (os.getpid(), time.time() - start_time, trial_key))
        sys.stdout.flush()

    if plot:
        if pop_order is None:
            pop_order = sorted(list(p_pos_dict.keys()))

        if compressed_plot_format:
            fig, axes = plt.subplots(2, len(pop_order), figsize=(2.2 * len(pop_order), 4.5))
        else:
            fig, axes = plt.subplots(2, len(pop_order), figsize=(4. * len(pop_order), 7.))
        if decode_duration > 1000.:
            this_binned_t_edges = decode_binned_t_edges / 1000.
        else:
            this_binned_t_edges = decode_binned_t_edges
        decoded_x_mesh, decoded_y_mesh = \
            np.meshgrid(this_binned_t_edges, template_binned_t_edges / template_duration)
        this_cmap = copy.copy(plt.get_cmap('binary'))
        this_cmap.set_bad(this_cmap(0.))
        for col, pop_name in enumerate(pop_order):
            p_pos = p_pos_dict[pop_name]
            axes[1][col].pcolormesh(decoded_x_mesh, decoded_y_mesh, p_pos, vmin=0., edgecolors='face', cmap=this_cmap,
                                    rasterized=True, antialiased=True)
            if decode_duration > 1000.:
                axes[1][col].set_xlabel('Time (s)')
            else:
                axes[1][col].set_xlabel('Time (ms)')
            axes[1][col].set_ylim((1., 0.))
            axes[1][col].set_xlim((this_binned_t_edges[0], this_binned_t_edges[-1]))
            if not compressed_plot_format:
                axes[1][col].set_ylabel('Decoded position')

            for i, gid in enumerate(sorted_gid_dict[pop_name]):
                this_spike_times = decode_spike_times_dict[pop_name][gid]
                if decode_duration > 1000.:
                    axes[0][col].scatter(this_spike_times / 1000., np.ones_like(this_spike_times) * i + 0.5, c='k',
                                         s=0.01, rasterized=True)
                else:
                    axes[0][col].scatter(this_spike_times, np.ones_like(this_spike_times) * i + 0.5, c='k',
                                         s=0.01, rasterized=True)
            # axes[0][col].set_xlabel('Time (s)')
            axes[0][col].set_ylim((len(sorted_gid_dict[pop_name]), 0))
            axes[0][col].set_xlim((this_binned_t_edges[0], this_binned_t_edges[-1]))
            if not compressed_plot_format:
                axes[0][col].set_ylabel('Sorted Cell ID')
            if label_dict is not None:
                label = label_dict[pop_name]
            else:
                label = pop_name
            axes[0][col].set_title(label, fontsize=mpl.rcParams['font.size'])
        if compressed_plot_format:
            axes[0][0].set_ylabel('Sorted Cell ID')
            axes[1][0].set_ylabel('Decoded position')
        fig.suptitle('Trial # %s' % trial_key, y=0.99, fontsize=mpl.rcParams['font.size'])
        fig.tight_layout()
        if compressed_plot_format:
            fig.subplots_adjust(wspace=0.5, hspace=0.45, top=0.87)
        else:
            fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9)

    return decoded_pos_dict


def decode_data(decode_data_file_path, decode_trial_keys, decode_group_key, export_data_key, decode_bin_dur,
                decode_duration, template_bin_dur, template_duration, export=False, temp_export=True, pop_order=None,
                label_dict=None, disp=True, plot=False, plot_trial_keys=None, compressed_plot_format=False):
    """

    :param decode_data_file_path: str (path)
    :param decode_trial_keys: list of str
    :param decode_group_key: str
    :param export_data_key: str
    :param decode_bin_dur: float (ms)
    :param decode_duration: float (ms)
    :param template_bin_dur: float (ms)
    :param template_duration: float (ms)
    :param export: bool
    :param temp_export: bool
    :param pop_order: list of str; order of populations for plot legend
    :param label_dict: dict; {pop_name: label}
    :param disp: bool
    :param plot: bool
    :param plot_trial_keys: list of str
    :param compressed_plot_format: bool
    :return: dict: {pop_name: 2D array}
    """
    num_trials = len(decode_trial_keys)
    if not plot or plot_trial_keys is None or len(plot_trial_keys) == 0:
        plot_list = [False] * num_trials
    else:
        plot_list = []
        for trial_key in decode_trial_keys:
            if trial_key in plot_trial_keys:
                plot_list.append(True)
            else:
                plot_list.append(False)
    sequences = [[decode_data_file_path] * num_trials, decode_trial_keys, [decode_group_key] * num_trials,
                 [export_data_key] * num_trials, [decode_bin_dur] * num_trials, [decode_duration] * num_trials,
                 [template_bin_dur] * num_trials, [template_duration] * num_trials, [temp_export] * num_trials,
                 [pop_order] * num_trials, [label_dict] * num_trials, [disp] * num_trials, plot_list,
                 [compressed_plot_format] * num_trials]
    context.interface.map(decode_single_trial_helper, *sequences)

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
            with h5py.File(decode_data_file_path, 'a') as f:
                group = get_h5py_group(f, [export_data_key, group_key, shared_context_key], create=True)
                subgroup = group.create_group('decoded_pos_matrix')
                for pop_name in decoded_pos_matrix_dict:
                    subgroup.create_dataset(pop_name, data=decoded_pos_matrix_dict[pop_name], compression='gzip')
            if disp:
                print('decode_data: pid: %i; exporting to decode_data_file_path: %s took %.1f s' %
                      (os.getpid(), decode_data_file_path, time.time() - start_time))

        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)

    return decoded_pos_matrix_dict


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
