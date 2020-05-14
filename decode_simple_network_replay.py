from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--run-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--replay-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--run-data-key", type=str, default='0')
@click.option("--num-replay-events", type=int, default=None)
@click.option("--window-dur", type=float, default=20.)
@click.option("--step-dur", type=float, default=20.)
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, run_data_file_path, replay_data_file_path, run_data_key, num_replay_events, window_dur, step_dur, export,
         output_dir, export_file_path, label, interactive, verbose, plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param run_data_file_path: str (path)
    :param replay_data_file_path: str (path)
    :param run_data_key: str
    :param num_events: int
    :param window_dur: float
    :param step_dur: float
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
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

    config_parallel_interface(__file__, output_dir=output_dir, export=export, export_file_path=export_file_path,
                              label=label, disp=context.disp, interface=context.interface, verbose=verbose, plot=plot,
                              debug=debug, **kwargs)

    args = context.interface.execute(get_replay_data_keys)
    num_replay_events = len(args[0])
    sequences = args + [[context.export] * num_replay_events] + [[context.plot] * num_replay_events]
    # decoded_position_package_list = context.interface.map(get_decoded_position_offline_replay, *sequences)

    if context.debug:
        print(args)

    if context.export:
        pass
    sys.stdout.flush()
    time.sleep(.1)

    if context.plot:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


def get_replay_data_keys():

    replay_data_group_key = 'simple_network_exported_data'
    with h5py.File(context.replay_data_file_path, 'r') as f:
        available_data_keys = \
            [data_key for data_key in f if data_key != 'shared_context' and replay_data_group_key in f[data_key]]
        if context.num_replay_events is None:
            replay_data_keys = available_data_keys
        else:
            replay_data_keys = \
                list(np.random.choice(available_data_keys, min(context.num_replay_events, len(available_data_keys)),
                                 replace=False))

    return [replay_data_keys]


def config_worker():

    start_time = time.time()

    if context.comm.rank == 0:
        run_binned_t, run_firing_rates_matrix_dict, sorted_gid_dict = \
            context.interface.execute(get_run_data_from_file, context.run_data_file_path, context.run_data_key)
    else:
        run_binned_t = None
        run_firing_rates_matrix_dict = None
        sorted_gid_dict = None
    context.comm.bcast(run_binned_t, root=0)
    context.comm.bcast(run_firing_rates_matrix_dict, root=0)
    context.comm.bcast(sorted_gid_dict, root=0)

    if context.comm.rank == 0:
        if not os.path.isfile(context.replay_data_file_path):
            raise IOError('decode_simple_network_replay: invalid replay data file path: %s' %
                          context.replay_data_file_path)

        with h5py.File(context.replay_data_file_path, 'r') as f:
            replay_input_t = f['shared_context']['full_binned_t'][:]
            replay_valid_t = f['shared_context']['buffered_binned_t'][:]
        valid_indexes = np.where((replay_input_t >= replay_valid_t[0]) & (replay_input_t <= replay_valid_t[-1]))[0]
        binned_dt = replay_input_t[1] - replay_input_t[0]
    else:
        replay_valid_t = None
        valid_indexes = None
        binned_dt = None
    context.comm.bcast(replay_valid_t, root=0)
    context.comm.bcast(valid_indexes, root=0)
    context.comm.bcast(binned_dt, root=0)

    window_dur = context.window_dur
    step_dur = context.step_dur

    align_to_t = 0.
    half_window_bins = int(window_dur // binned_dt // 2)
    window_bins = int(2 * half_window_bins + 1)
    window_dur = window_bins * binned_dt

    step_bins = step_dur // binned_dt
    step_dur = step_bins * binned_dt
    half_step_dur = step_dur / 2.

    # if possible, include a bin centered on time zero.
    binned_t_center_indexes = []
    this_center_index = np.where(replay_valid_t >= align_to_t)[0]
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
    while this_center_index < len(replay_valid_t) - half_window_bins:
        binned_t_center_indexes.append(this_center_index)
        this_center_index += step_bins

    binned_t_center_indexes = np.array(binned_t_center_indexes, dtype='int')
    decode_binned_t = replay_valid_t[binned_t_center_indexes]
    replay_x_mesh, replay_y_mesh = np.meshgrid(decode_binned_t - half_step_dur, run_binned_t)

    context.update(locals())


def analyze_network_output(network, model_id=None, export=False, plot=False):
    """

    :param network: :class:'SimpleNetwork'
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    full_rec_t = np.arange(-context.buffer - context.equilibrate, context.duration + context.buffer + context.dt / 2.,
                           context.dt)
    buffered_rec_t = np.arange(-context.buffer, context.duration + context.buffer + context.dt / 2., context.dt)
    rec_t = np.arange(0., context.duration, context.dt)
    full_binned_t = np.arange(-context.buffer-context.equilibrate,
                              context.duration + context.buffer + context.binned_dt / 2., context.binned_dt)
    buffered_binned_t = np.arange(-context.buffer, context.duration + context.buffer + context.binned_dt / 2.,
                                  context.binned_dt)
    binned_t = np.arange(0., context.duration + context.binned_dt / 2., context.binned_dt)

    full_spike_times_dict = network.get_spike_times_dict()
    buffered_firing_rates_dict = \
        infer_firing_rates_baks(full_spike_times_dict, buffered_binned_t, alpha=context.baks_alpha,
                                beta=context.baks_beta, pad_dur=context.baks_pad_dur,
                                wrap_around=context.baks_wrap_around)
    full_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, full_binned_t)
    buffered_firing_rates_from_binned_spike_count_dict = \
        infer_firing_rates_from_spike_count(full_binned_spike_count_dict, input_t=full_binned_t,
                                            output_t=buffered_binned_t, align_to_t=0., window_dur=40.,
                                            step_dur=context.binned_dt, smooth_dur=20.,
                                            debug=context.debug and context.comm.rank == 0)
    gathered_full_spike_times_dict_list = context.comm.gather(full_spike_times_dict, root=0)
    gathered_buffered_firing_rates_dict_list = context.comm.gather(buffered_firing_rates_dict, root=0)
    gathered_full_binned_spike_count_dict_list = context.comm.gather(full_binned_spike_count_dict, root=0)
    gathered_buffered_firing_rates_from_binned_spike_count_dict_list = \
        context.comm.gather(buffered_firing_rates_from_binned_spike_count_dict, root=0)

    full_voltage_rec_dict = network.get_voltage_rec_dict()
    voltages_exceed_threshold = check_voltages_exceed_threshold(full_voltage_rec_dict, input_t=full_rec_t,
                                                                valid_t=buffered_rec_t,
                                                                pop_cell_types=context.pop_cell_types)
    voltages_exceed_threshold_list = context.comm.gather(voltages_exceed_threshold, root=0)

    if plot or export:
        bcast_subset_voltage_rec_gids = dict()
        if context.comm.rank == 0:
            for pop_name in (pop_name for pop_name in context.pop_cell_types
                             if context.pop_cell_types[pop_name] != 'input'):
                gather_gids = random.sample(range(*context.pop_gid_ranges[pop_name]),
                                            min(context.max_num_cells_export_voltage_rec, context.pop_sizes[pop_name]))
                bcast_subset_voltage_rec_gids[pop_name] = set(gather_gids)
        bcast_subset_voltage_rec_gids = context.comm.bcast(bcast_subset_voltage_rec_gids, root=0)

        subset_full_voltage_rec_dict = dict()
        for pop_name in bcast_subset_voltage_rec_gids:
            if pop_name in full_voltage_rec_dict:
                for gid in full_voltage_rec_dict[pop_name]:
                    if gid in bcast_subset_voltage_rec_gids[pop_name]:
                        if pop_name not in subset_full_voltage_rec_dict:
                            subset_full_voltage_rec_dict[pop_name] = dict()
                        subset_full_voltage_rec_dict[pop_name][gid] = full_voltage_rec_dict[pop_name][gid]
        gathered_subset_full_voltage_rec_dict_list = context.comm.gather(subset_full_voltage_rec_dict, root=0)

        connectivity_dict = network.get_connectivity_dict()
        connectivity_dict = context.comm.gather(connectivity_dict, root=0)

        connection_target_gid_dict, connection_weights_dict = network.get_connection_weights()
        gathered_connection_target_gid_dict_list = context.comm.gather(connection_target_gid_dict, root=0)
        gathered_connection_weights_dict_list = context.comm.gather(connection_weights_dict, root=0)

    if context.debug and context.verbose > 0 and context.comm.rank == 0:
        print('optimize_simple_network_replay: pid: %i; gathering data across ranks took %.2f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()
        current_time = time.time()

    if context.comm.rank == 0:
        full_spike_times_dict = merge_list_of_dict(gathered_full_spike_times_dict_list)
        buffered_firing_rates_dict = merge_list_of_dict(gathered_buffered_firing_rates_dict_list)
        full_binned_spike_count_dict = merge_list_of_dict(gathered_full_binned_spike_count_dict_list)
        buffered_firing_rates_from_binned_spike_count_dict = \
            merge_list_of_dict(gathered_buffered_firing_rates_from_binned_spike_count_dict_list)

        if plot or export:
            subset_full_voltage_rec_dict = merge_list_of_dict(gathered_subset_full_voltage_rec_dict_list)
            connectivity_dict = merge_list_of_dict(connectivity_dict)
            connection_weights_dict = \
                merge_connection_weights_dicts(gathered_connection_target_gid_dict_list,
                                               gathered_connection_weights_dict_list)

        if context.debug and context.verbose > 0:
            print('optimize_simple_network: pid: %i; merging data structures for analysis took %.2f s' %
                  (os.getpid(), time.time() - current_time))
            current_time = time.time()
            sys.stdout.flush()

        full_pop_mean_rate_from_binned_spike_count_dict = \
            get_pop_mean_rate_from_binned_spike_count(full_binned_spike_count_dict, dt=context.binned_dt)
        mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict = \
            get_pop_activity_stats(buffered_firing_rates_from_binned_spike_count_dict, input_t=buffered_binned_t,
                                   valid_t=buffered_binned_t, threshold=context.active_rate_threshold, plot=plot)
        filtered_mean_rate_dict, filter_envelope_dict, filter_envelope_ratio_dict, centroid_freq_dict, \
            freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(full_pop_mean_rate_from_binned_spike_count_dict,
                                                   context.filter_bands, input_t=full_binned_t,
                                                   valid_t=binned_t, output_t=binned_t,
                                                   plot=plot, verbose=context.verbose > 1)

        if context.debug and context.verbose > 0:
            print('optimize_simple_network: pid: %i; additional data analysis took %.2f s' %
                  (os.getpid(), time.time() - current_time))
            sys.stdout.flush()

        if plot:
            """
            plot_inferred_spike_rates(full_spike_times_dict, buffered_firing_rates_from_binned_spike_count_dict,
                                      input_t=buffered_binned_t, valid_t=buffered_binned_t,
                                      active_rate_threshold=context.active_rate_threshold)
            plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t,
                                spike_times_dict=full_spike_times_dict)
            plot_weight_matrix(connection_weights_dict, pop_gid_ranges=context.pop_gid_ranges,
                               tuning_peak_locs=context.tuning_peak_locs)
            """
            plot_firing_rate_heatmaps(buffered_firing_rates_from_binned_spike_count_dict, input_t=buffered_binned_t,
                                      valid_t=buffered_binned_t, tuning_peak_locs=context.tuning_peak_locs)
            plot_firing_rate_heatmaps(full_binned_spike_count_dict, input_t=full_binned_t,
                                      valid_t=buffered_binned_t, tuning_peak_locs=context.tuning_peak_locs)
            if context.connectivity_type == 'gaussian':
                plot_2D_connection_distance(context.pop_syn_proportions, context.pop_cell_positions, connectivity_dict)

        if any(voltages_exceed_threshold_list):
            voltages_exceed_threshold = True
            if context.verbose > 0:
                print('optimize_simple_network: pid: %i; model_id: %i; model failed - mean membrane voltage in some '
                      'Izhi cells exceeds spike threshold' % (os.getpid(), model_id))
                sys.stdout.flush()
        else:
            voltages_exceed_threshold = False

        if export:
            current_time = time.time()
            with h5py.File(context.temp_output_path, 'a') as f:
                shared_context_key = 'shared_context'
                if model_id == 0 and 'shared_context' not in f:
                    group = f.create_group('shared_context')
                    group.create_dataset('param_names', data=np.array(context.param_names, dtype='S'),
                                         compression='gzip')
                    group.create_dataset('x0', data=context.x0_array, compression='gzip')
                    set_h5py_attr(group.attrs, 'connectivity_type', context.connectivity_type)
                    group.attrs['active_rate_threshold'] = context.active_rate_threshold
                    subgroup = group.create_group('pop_gid_ranges')
                    for pop_name in context.pop_gid_ranges:
                        subgroup.create_dataset(pop_name, data=context.pop_gid_ranges[pop_name])
                    group.create_dataset('full_binned_t', data=full_binned_t, compression='gzip')
                    group.create_dataset('buffered_binned_t', data=buffered_binned_t, compression='gzip')
                    group.create_dataset('binned_t', data=binned_t, compression='gzip')
                    subgroup = group.create_group('filter_bands')
                    for filter, band in viewitems(context.filter_bands):
                        subgroup.create_dataset(filter, data=band)
                    group.create_dataset('full_rec_t', data=full_rec_t, compression='gzip')
                    group.create_dataset('buffered_rec_t', data=buffered_rec_t, compression='gzip')
                    group.create_dataset('rec_t', data=rec_t, compression='gzip')
                    subgroup = group.create_group('connection_weights')
                    for target_pop_name in connection_weights_dict:
                        subgroup.create_group(target_pop_name)
                        for source_pop_name in connection_weights_dict[target_pop_name]:
                            subgroup[target_pop_name].create_dataset(
                                source_pop_name, data=connection_weights_dict[target_pop_name][source_pop_name],
                                compression='gzip')
                    if len(context.tuning_peak_locs) > 0:
                        subgroup = group.create_group('tuning_peak_locs')
                        for pop_name in context.tuning_peak_locs:
                            if len(context.tuning_peak_locs[pop_name]) > 0:
                                data_group = subgroup.create_group(pop_name)
                                target_gids = np.array(list(context.tuning_peak_locs[pop_name].keys()))
                                peak_locs = np.array(list(context.tuning_peak_locs[pop_name].values()))
                                data_group.create_dataset('target_gids', data=target_gids, compression='gzip')
                                data_group.create_dataset('peak_locs', data=peak_locs, compression='gzip')
                    subgroup = group.create_group('connectivity')
                    for target_pop_name in connectivity_dict:
                        subgroup.create_group(target_pop_name)
                        for target_gid in connectivity_dict[target_pop_name]:
                            data_group = subgroup[target_pop_name].create_group(str(target_gid))
                            for source_pop_name in connectivity_dict[target_pop_name][target_gid]:
                                data_group.create_dataset(source_pop_name,
                                                          data=connectivity_dict[target_pop_name][target_gid][
                                                              source_pop_name])
                    subgroup = group.create_group('pop_syn_proportions')
                    for target_pop_name in context.pop_syn_proportions:
                        subgroup.create_group(target_pop_name)
                        for syn_type in context.pop_syn_proportions[target_pop_name]:
                            data_group = subgroup[target_pop_name].create_group(syn_type)
                            source_pop_names = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].keys()), dtype='S')
                            syn_proportions = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].values()))
                            data_group.create_dataset('source_pop_names', data=source_pop_names)
                            data_group.create_dataset('syn_proportions', data=syn_proportions)
                    subgroup = group.create_group('pop_cell_positions')
                    for pop_name in context.pop_cell_positions:
                        data_group = subgroup.create_group(pop_name)
                        gids = np.array(list(context.pop_cell_positions[pop_name].keys()))
                        positions = np.array(list(context.pop_cell_positions[pop_name].values()))
                        data_group.create_dataset('gids', data=gids, compression='gzip')
                        data_group.create_dataset('positions', data=positions, compression='gzip')
                exported_data_key = 'simple_network_exported_data'
                group = get_h5py_group(f, [model_id, exported_data_key], create=True)
                set_h5py_attr(group.attrs, 'voltages_exceed_threshold', voltages_exceed_threshold)
                subgroup = group.create_group('full_spike_times')
                for pop_name in full_spike_times_dict:
                    subgroup.create_group(pop_name)
                    for gid in full_spike_times_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=full_spike_times_dict[pop_name][gid], compression='gzip')
                subgroup = group.create_group('buffered_firing_rates_from_binned_spike_count')
                for pop_name in buffered_firing_rates_from_binned_spike_count_dict:
                    subgroup.create_group(pop_name)
                    for gid in buffered_firing_rates_from_binned_spike_count_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=buffered_firing_rates_from_binned_spike_count_dict[pop_name][gid],
                            compression='gzip')
                subgroup = group.create_group('full_binned_spike_count')
                for pop_name in full_binned_spike_count_dict:
                    subgroup.create_group(pop_name)
                    for gid in full_binned_spike_count_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=full_binned_spike_count_dict[pop_name][gid], compression='gzip')
                subgroup = group.create_group('subset_full_voltage_recs')
                for pop_name in subset_full_voltage_rec_dict:
                    subgroup.create_group(pop_name)
                    for gid in subset_full_voltage_rec_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=subset_full_voltage_rec_dict[pop_name][gid], compression='gzip')

            print('optimize_simple_network: pid: %i; model_id: %i; exporting data to file: %s took %.2f s' %
                  (os.getpid(), model_id, context.temp_output_path, time.time() - current_time))
            sys.stdout.flush()

        if context.interactive:
            context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
