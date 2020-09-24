from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/optimize_simple_network_J_alt_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--simulate", type=bool, default=True)
@click.pass_context
def main(cli, config_file_path, export, output_dir, export_file_path, label, interactive, verbose, plot, debug,
         simulate):
    """

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path)
    :param export: bool
    :param output_dir: str
    :param export_file_path: str
    :param label: str
    :param interactive: bool
    :param verbose: int
    :param plot: bool
    :param debug: bool
    :param simulate: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.disp = verbose > 0

    if 'procs_per_worker' not in kwargs:
        kwargs['procs_per_worker'] = int(MPI.COMM_WORLD.size)

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=context.disp)
    context.interface.ensure_controller()
    config_optimize_interactive(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                                export_file_path=export_file_path, label=label, disp=context.disp,
                                interface=context.interface, verbose=verbose, plot=plot, debug=debug, **kwargs)
    if simulate:
        run_tests()

    if context.plot:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()


def run_tests():

    model_id = 0
    if 'model_key' in context() and context.model_key is not None:
        model_label = context.model_key
    else:
        model_label = 'test'

    features = context.interface.execute(compute_features_run, context.x0_array, model_id, context.export)

    args = context.interface.execute(get_args_static_replay_event_ids)
    group_size = len(args[0])
    sequences = [[context.x0_array] * group_size] + args + [[model_id] * group_size] + [[context.export] * group_size]
    primitives = context.interface.map(compute_features_replay, *sequences)
    this_features = context.interface.execute(filter_features_replay, primitives, features, model_id, context.export)
    features.update(this_features)
    sys.stdout.flush()
    time.sleep(0.1)

    if len(features) > 0:
        features, objectives = context.interface.execute(get_objectives, features, model_id, context.export)
    else:
        objectives = dict()

    if context.export:
        merge_exported_data(context, param_arrays=[context.x0_array],
                            model_ids=[model_id], model_labels=[model_label], features=[features],
                            objectives=[objectives], export_file_path=context.export_file_path,
                            verbose=context.verbose > 1)
    sys.stdout.flush()
    print('model_id: %i; model_labels: %s' % (model_id, model_label))
    print('params:')
    pprint.pprint(context.x0_dict)
    print('features:')
    pprint.pprint(features)
    print('objectives:')
    pprint.pprint(objectives)
    sys.stdout.flush()
    time.sleep(.1)

    context.update(locals())


def config_worker():

    if 'plot' not in context():
        context.plot = False
    if 'verbose' not in context():
        context.verbose = 1
    else:
        context.verbose = int(context.verbose)
    if 'debug' not in context():
        context.debug = False

    context.pc = h.ParallelContext()

    start_time = time.time()

    # {'pop_name': str}
    if 'pop_cell_types' not in context() or context.pop_cell_types is None:
        context.pop_cell_types = {'FF': 'input', 'E': 'IB', 'I': 'FS'}

    # {'pop_name': int}
    if 'pop_sizes' not in context() or context.pop_sizes is None:
        raise RuntimeError('optimize_simple_network: pop_sizes must be specified in the config.yaml')
    for pop_name in context.pop_sizes:
        context.pop_sizes[pop_name] = int(context.pop_sizes[pop_name])

    total_cells = np.sum(list(context.pop_sizes.values()))

    pop_gid_ranges = get_pop_gid_ranges(context.pop_sizes)
    
    if 'input_mean_rates_run' not in context():
        input_mean_rates_run = None
    if 'input_min_rates_run' not in context():
        input_min_rates_run = None
    if 'input_max_rates_run' not in context():
        input_max_rates_run = None
    if 'input_offline_min_rates' not in context():
        input_offline_min_rates = None
    if 'input_offline_mean_rates' not in context():
        input_offline_mean_rates = None
    if 'input_offline_fraction_active' not in context():
        input_offline_fraction_active = None
    if 'input_norm_tuning_widths' not in context():
        input_norm_tuning_widths = None
    if 'replay_trial_offset' not in context():
        context.replay_trial_offset = 0
    else:
        context.replay_trial_offset = int(context.replay_trial_offset)
    if 'num_replay_trials' not in context():
        context.num_replay_trials = 1
    else:
        context.num_replay_trials = int(context.num_replay_trials)

    if 'network_id' not in context():
        context.network_id = 0
    else:
        context.network_id = int(context.network_id)
    if 'network_instance' not in context():
        context.network_instance = 0
    else:
        context.network_instance = int(context.network_instance)
    if 'run_trial' not in context():
        context.run_trial = 0
    else:
        context.run_trial = int(context.run_trial)

    context.run_stim_type_seed = 0
    context.replay_stim_type_seed = 1
    context.base_seed = [context.network_id, context.network_instance]
    context.connection_seed = context.base_seed + [1]
    context.weights_seed = context.base_seed + [3]
    context.location_seed = context.base_seed + [4]
    context.tuning_seed = context.base_seed + [5]
    context.selection_seed = context.base_seed + [6]

    connection_weights_mean = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    connection_weights_norm_sigma = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    if 'connection_weight_distribution_types' not in context() or context.connection_weight_distribution_types is None:
        context.connection_weight_distribution_types = dict()
    if 'structured_weight_params' not in context() or context.structured_weight_params is None:
        context.structured_weight_params = dict()
        structured_weights = False
    else:
        structured_weights = True

    syn_mech_params = defaultdict(lambda: defaultdict(dict))
    if 'default_syn_mech_params' in context() and context.default_syn_mech_params is not None:
        for target_pop_name in context.default_syn_mech_params:
            for source_pop_name in context.default_syn_mech_params[target_pop_name]:
                for param_name in context.default_syn_mech_params[target_pop_name][source_pop_name]:
                    syn_mech_params[target_pop_name][source_pop_name][param_name] = \
                        context.default_syn_mech_params[target_pop_name][source_pop_name][param_name]

    delay = 1.  # ms
    equilibrate = 250.  # ms
    run_buffer = 500.  # ms
    run_duration = 3000.  # ms
    replay_buffer = 150.  # ms
    replay_duration = 200.  # ms
    replay_stim_edge_duration = (150., 25.)  # ms

    if any(np.array(replay_stim_edge_duration) > replay_buffer):
        raise RuntimeError('Offline stimulus edge duration must be less than simulation buffer duration.')

    tuning_duration = 3000.  # ms
    run_tstop = int(equilibrate + run_duration + 2. * run_buffer)  # ms
    replay_tstop = int(equilibrate + replay_duration + 2. * replay_buffer)  # ms
    dt = 0.025
    fine_binned_dt = 1.  # ms
    coarse_binned_dt = 20.  # ms
    active_rate_threshold = 1.  # Hz
    baks_alpha = 4.7725100028345535
    baks_beta = 0.41969058927343522
    run_baks_pad_dur = run_duration  # ms
    run_baks_wrap_around = True
    replay_baks_pad_dur = 0.  # ms
    replay_baks_wrap_around = False
    track_wrap_around = True
    run_filter_bands = {'Theta': [4., 10.], 'Gamma': [30., 100.], 'Ripple': [100., 200.]}
    replay_filter_bands = {'Ripple': [100., 200.]}
    max_num_cells_export_voltage_rec = 500

    if context.comm.rank == 0:
        if any([this_input_type == 'gaussian' for this_input_type in
                context.input_types.values()]) or structured_weights:
            local_np_random = np.random.default_rng(seed=context.tuning_seed)
        tuning_peak_locs = dict()  # {'pop_name': {'gid': float} }

        for pop_name in sorted(list(context.input_types.keys())):
            if pop_name not in context.pop_cell_types or context.pop_cell_types[pop_name] != 'input':
                raise RuntimeError('optimize_simple_network: %s not specified as an input population' % pop_name)
            if context.input_types[pop_name] == 'gaussian':
                if pop_name not in tuning_peak_locs:
                    tuning_peak_locs[pop_name] = dict()
                peak_locs_array = \
                    np.linspace(0., tuning_duration, context.pop_sizes[pop_name], endpoint=False)
                local_np_random.shuffle(peak_locs_array)
                for peak_loc, gid in zip(peak_locs_array, range(pop_gid_ranges[pop_name][0],
                                                                pop_gid_ranges[pop_name][1])):
                    tuning_peak_locs[pop_name][gid] = peak_loc

        """
        TODO: collect network.input_pop_firing_rates from each rank and plot on rank 0 
        if context.debug and context.plot:
            fig, axes = plt.subplots()
            for gid in range(pop_gid_ranges[pop_name][0],
                             pop_gid_ranges[pop_name][1])[::int(context.pop_sizes[pop_name] / 25)]:
                axes.plot(this_stim_t - buffer, input_pop_firing_rates[pop_name][gid])
            mean_input = np.mean(list(input_pop_firing_rates[pop_name].values()), axis=0)
            axes.plot(this_stim_t - buffer, mean_input, c='k', linewidth=2.)
            axes.set_ylabel('Firing rate (Hz)')
            axes.set_xlabel('Time (ms)')
            clean_axes(axes)
            fig.show()
        """

        for target_pop_name in sorted(list(context.structured_weight_params.keys())):
            if target_pop_name not in tuning_peak_locs:
                tuning_peak_locs[target_pop_name] = dict()
            if 'pop_over_representation' in context.structured_weight_params[target_pop_name]:
                over_rep_loc = context.structured_weight_params[target_pop_name]['pop_over_representation']['loc']
                over_rep_width = context.structured_weight_params[target_pop_name]['pop_over_representation']['width']
                over_rep_depth = context.structured_weight_params[target_pop_name]['pop_over_representation']['depth']
                over_rep_sigma = over_rep_width * tuning_duration / 3. / np.sqrt(2.)
                available_peak_locs, p_peak_locs = \
                    get_gaussian_prob_peak_locs(tuning_duration, context.pop_sizes[target_pop_name],
                                                over_rep_loc * tuning_duration, over_rep_sigma, over_rep_depth,
                                                resolution=2, wrap_around=track_wrap_around)
                peak_locs_array = local_np_random.choice(available_peak_locs, size=context.pop_sizes[target_pop_name],
                                                         p=p_peak_locs, replace=False)
            else:
                peak_locs_array = \
                    np.linspace(0., tuning_duration, context.pop_sizes[target_pop_name], endpoint=False)
            local_np_random.shuffle(peak_locs_array)
            for peak_loc, target_gid in zip(peak_locs_array,
                                            range(pop_gid_ranges[target_pop_name][0],
                                                  pop_gid_ranges[target_pop_name][1])):
                tuning_peak_locs[target_pop_name][target_gid] = peak_loc
    else:
        tuning_peak_locs = None
    tuning_peak_locs = context.comm.bcast(tuning_peak_locs, root=0)

    if context.connectivity_type == 'gaussian':
        pop_axon_extents = {'FF': 1., 'E': 1., 'I': 1.}
        if 'spatial_dim' not in context():
            raise RuntimeError('optimize_simple_network: missing spatial_dim parameter required for gaussian '
                               'connectivity not found')

        if context.comm.rank == 0:
            local_np_random = np.random.default_rng(seed=context.location_seed)
            pop_cell_positions = dict()
            for pop_name in sorted(list(pop_gid_ranges.keys())):
                for gid in range(pop_gid_ranges[pop_name][0], pop_gid_ranges[pop_name][1]):
                    if pop_name not in pop_cell_positions:
                        pop_cell_positions[pop_name] = dict()
                    pop_cell_positions[pop_name][gid] = local_np_random.uniform(-1., 1., size=context.spatial_dim)
        else:
            pop_cell_positions = None
        pop_cell_positions = context.comm.bcast(pop_cell_positions, root=0)
    else:
        pop_cell_positions = dict()

    context.update(locals())
    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; worker initialization took %.2f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()


def get_args_static_replay_event_ids():

    return [list(range(context.replay_trial_offset, context.replay_trial_offset + context.num_replay_trials)),
            list(range(context.replay_trial_offset, context.replay_trial_offset + context.num_replay_trials))]


def update_context(x, local_context=None):
    """

    :param x: array
    :param local_context: :class:'Context'
    """
    if local_context is None:
        local_context = context
    x_dict = param_array_to_dict(x, local_context.param_names)
    
    local_context.syn_mech_params['E']['FF']['tau_offset'] = x_dict['E_E_tau_offset']
    local_context.syn_mech_params['E']['E']['tau_offset'] = x_dict['E_E_tau_offset']
    local_context.syn_mech_params['E']['I']['tau_offset'] = x_dict['E_I_tau_offset']
    local_context.syn_mech_params['I']['FF']['tau_offset'] = x_dict['I_E_tau_offset']
    local_context.syn_mech_params['I']['E']['tau_offset'] = x_dict['I_E_tau_offset']
    local_context.syn_mech_params['I']['I']['tau_offset'] = x_dict['I_I_tau_offset']

    local_context.connection_weights_mean['E']['FF'] = x_dict['E_FF_mean_weight']
    local_context.connection_weights_mean['E']['E'] = x_dict['E_E_mean_weight']
    local_context.connection_weights_mean['E']['I'] = x_dict['E_I_mean_weight']
    local_context.connection_weights_mean['I']['FF'] = x_dict['I_FF_mean_weight']
    local_context.connection_weights_mean['I']['E'] = x_dict['I_E_mean_weight']
    local_context.connection_weights_mean['I']['I'] = x_dict['I_I_mean_weight']

    local_context.connection_weights_norm_sigma['E']['FF'] = x_dict['E_FF_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['E']['E'] = x_dict['E_E_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['E']['I'] = x_dict['E_I_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['FF'] = x_dict['I_FF_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['E'] = x_dict['I_E_weight_norm_sigma']
    local_context.connection_weights_norm_sigma['I']['I'] = x_dict['I_I_weight_norm_sigma']

    context.pop_syn_counts = dict()
    context.pop_syn_counts['E'] = int(context.total_cells * x_dict['E_norm_syn_count'])
    context.pop_syn_counts['I'] = int(context.total_cells * x_dict['I_norm_syn_count'])

    # {'target_pop_name': {'syn_type: {'source_pop_name': float} } }
    local_context.pop_syn_proportions = defaultdict(lambda: defaultdict(dict))
    local_context.pop_syn_proportions['E']['E']['FF'] = x_dict['E_E_syn_proportion'] * x_dict['E_E_FF_syn_proportion']
    local_context.pop_syn_proportions['E']['E']['E'] = x_dict['E_E_syn_proportion'] * \
                                                       (1. - x_dict['E_E_FF_syn_proportion'])
    local_context.pop_syn_proportions['E']['I']['I'] = 1. - x_dict['E_E_syn_proportion']
    local_context.pop_syn_proportions['I']['E']['FF'] = x_dict['I_E_syn_proportion'] * x_dict['I_E_FF_syn_proportion']
    local_context.pop_syn_proportions['I']['E']['E'] = x_dict['I_E_syn_proportion'] * \
                                                       (1. - x_dict['I_E_FF_syn_proportion'])
    local_context.pop_syn_proportions['I']['I']['I'] = 1. - x_dict['I_E_syn_proportion']

    if local_context.structured_weights:
        for target_pop_name in local_context.structured_weight_params:
            peak_delta_weight_param_name = '%s_peak_delta_weight' % target_pop_name
            if peak_delta_weight_param_name in x_dict:
                local_context.structured_weight_params[target_pop_name]['peak_delta_weight'] = \
                    x_dict[peak_delta_weight_param_name]


def analyze_network_output_run(network, model_id=None, export=False, plot=False):
    """

    :param network: :class:'SimpleNetwork'
    :param model_id: int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    context.filter_bands = context.run_filter_bands
    full_rec_t = np.arange(-context.buffer - context.equilibrate, context.duration + context.buffer + context.dt / 2.,
                           context.dt)
    buffered_rec_t = np.arange(-context.buffer, context.duration + context.buffer + context.dt / 2., context.dt)
    rec_t = np.arange(0., context.duration, context.dt)
    buffered_binned_t_edges = \
        np.arange(-context.buffer, context.duration + context.buffer + context.fine_binned_dt / 2.,
                  context.fine_binned_dt)
    buffered_binned_t = buffered_binned_t_edges[:-1] + context.fine_binned_dt / 2.
    binned_t_edges = np.arange(0., context.duration + context.fine_binned_dt / 2., context.fine_binned_dt)
    binned_t = binned_t_edges[:-1] + context.fine_binned_dt / 2.

    full_spike_times_dict = network.get_spike_times_dict()
    binned_firing_rates_dict = \
        infer_firing_rates_baks(full_spike_times_dict, binned_t_edges, alpha=context.baks_alpha,
                                beta=context.baks_beta, pad_dur=context.baks_pad_dur,
                                wrap_around=context.baks_wrap_around)
    buffered_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, buffered_binned_t_edges)

    gathered_full_spike_times_dict_list = context.comm.gather(full_spike_times_dict, root=0)
    gathered_binned_firing_rates_dict_list = context.comm.gather(binned_firing_rates_dict, root=0)
    gathered_buffered_binned_spike_count_dict_list = context.comm.gather(buffered_binned_spike_count_dict, root=0)

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
        print('optimize_simple_network: analyze_network_output_run: pid: %i; gathering data across ranks took %.2f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()
        current_time = time.time()

    if context.comm.rank == 0:
        full_spike_times_dict = merge_list_of_dict(gathered_full_spike_times_dict_list)
        binned_firing_rates_dict = merge_list_of_dict(gathered_binned_firing_rates_dict_list)
        buffered_binned_spike_count_dict = merge_list_of_dict(gathered_buffered_binned_spike_count_dict_list)

        if plot or export:
            subset_full_voltage_rec_dict = merge_list_of_dict(gathered_subset_full_voltage_rec_dict_list)
            connectivity_dict = merge_list_of_dict(connectivity_dict)
            connection_weights_dict = \
                merge_connection_weights_dicts(gathered_connection_target_gid_dict_list,
                                               gathered_connection_weights_dict_list)

        if context.debug and context.verbose > 0:
            print('optimize_simple_network: analyze_network_output_run: pid: %i; merging data structures for analysis '
                  'took %.2f s' %
                  (os.getpid(), time.time() - current_time))
            current_time = time.time()
            sys.stdout.flush()

        buffered_pop_mean_rate_from_binned_spike_count_dict = \
            get_pop_mean_rate_from_binned_spike_count(buffered_binned_spike_count_dict, dt=context.fine_binned_dt)
        mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict = \
            get_pop_activity_stats(binned_firing_rates_dict, input_t=binned_t_edges,
                                   threshold=context.active_rate_threshold, plot=plot)

        fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
        filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(buffered_pop_mean_rate_from_binned_spike_count_dict,
                                                   context.filter_bands, input_t=buffered_binned_t,
                                                   valid_t=buffered_binned_t, output_t=binned_t, pad=True,
                                                   plot=plot, verbose=context.verbose > 1)

        if context.debug:
            context.update(locals())
            return

        if context.debug and context.verbose > 0:
            print('optimize_simple_network: analyze_network_output_run: pid: %i; additional data analysis took %.2f s' %
                  (os.getpid(), time.time() - current_time))
            sys.stdout.flush()

        if plot:
            plot_inferred_spike_rates(full_spike_times_dict, binned_firing_rates_dict, input_t=binned_t_edges,
                                      active_rate_threshold=context.active_rate_threshold)
            plot_voltage_traces(subset_full_voltage_rec_dict, full_rec_t, valid_t=rec_t,
                                spike_times_dict=full_spike_times_dict)
            plot_weight_matrix(connection_weights_dict, pop_gid_ranges=context.pop_gid_ranges,
                               tuning_peak_locs=context.tuning_peak_locs)
            plot_firing_rate_heatmaps(binned_firing_rates_dict, input_t=binned_t_edges,
                                      tuning_peak_locs=context.tuning_peak_locs)
            if context.connectivity_type == 'gaussian':
                plot_2D_connection_distance(context.pop_syn_proportions, context.pop_cell_positions, connectivity_dict)

        if export:
            current_time = time.time()
            with h5py.File(context.temp_output_path, 'a') as f:
                run_group_key = 'simple_network_exported_run_data'
                group = get_h5py_group(f, [model_id, run_group_key], create=True)
                shared_context_key = 'shared_context'
                if shared_context_key not in group:
                    subgroup = group.create_group(shared_context_key)
                    subgroup.create_dataset('param_names', data=np.array(context.param_names, dtype='S'),
                                            compression='gzip')
                    subgroup.create_dataset('x0', data=context.x0_array, compression='gzip')
                    set_h5py_attr(subgroup.attrs, 'network_id', context.network_id)
                    set_h5py_attr(subgroup.attrs, 'network_instance', context.network_instance)
                    set_h5py_attr(subgroup.attrs, 'connectivity_type', context.connectivity_type)
                    set_h5py_attr(subgroup.attrs, 'duration', context.duration)
                    set_h5py_attr(subgroup.attrs, 'buffer', context.duration)
                    set_h5py_attr(subgroup.attrs, 'active_rate_threshold', context.active_rate_threshold)
                    set_h5py_attr(subgroup.attrs, 'baks_alpha', context.baks_alpha)
                    set_h5py_attr(subgroup.attrs, 'baks_beta', context.baks_beta)
                    set_h5py_attr(subgroup.attrs, 'baks_pad_dur', context.baks_pad_dur)
                    set_h5py_attr(subgroup.attrs, 'baks_wrap_around', context.baks_wrap_around)
                    data_group = subgroup.create_group('pop_gid_ranges')
                    for pop_name in context.pop_gid_ranges:
                        data_group.create_dataset(pop_name, data=context.pop_gid_ranges[pop_name])
                    data_group = subgroup.create_group('filter_bands')
                    for filter, band in viewitems(context.filter_bands):
                        data_group.create_dataset(filter, data=band)
                    data_group = subgroup.create_group('connection_weights')
                    for target_pop_name in connection_weights_dict:
                        data_group.create_group(target_pop_name)
                        for source_pop_name in connection_weights_dict[target_pop_name]:
                            data_group[target_pop_name].create_dataset(
                                source_pop_name, data=connection_weights_dict[target_pop_name][source_pop_name],
                                compression='gzip')
                    if len(context.tuning_peak_locs) > 0:
                        data_group = subgroup.create_group('tuning_peak_locs')
                        for pop_name in context.tuning_peak_locs:
                            if len(context.tuning_peak_locs[pop_name]) > 0:
                                data_group.create_group(pop_name)
                                target_gids = np.array(list(context.tuning_peak_locs[pop_name].keys()))
                                peak_locs = np.array(list(context.tuning_peak_locs[pop_name].values()))
                                data_group[pop_name].create_dataset('target_gids', data=target_gids, compression='gzip')
                                data_group[pop_name].create_dataset('peak_locs', data=peak_locs, compression='gzip')
                    data_group = subgroup.create_group('connectivity')
                    for target_pop_name in connectivity_dict:
                        data_group.create_group(target_pop_name)
                        for target_gid in connectivity_dict[target_pop_name]:
                            data_subgroup = data_group[target_pop_name].create_group(str(target_gid))
                            for source_pop_name in connectivity_dict[target_pop_name][target_gid]:
                                data_subgroup.create_dataset(source_pop_name,
                                                             data=connectivity_dict[target_pop_name][target_gid][
                                                                 source_pop_name])
                    data_group = subgroup.create_group('pop_syn_proportions')
                    for target_pop_name in context.pop_syn_proportions:
                        data_group.create_group(target_pop_name)
                        for syn_type in context.pop_syn_proportions[target_pop_name]:
                            data_subgroup = data_group[target_pop_name].create_group(syn_type)
                            source_pop_names = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].keys()), dtype='S')
                            syn_proportions = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].values()))
                            data_subgroup.create_dataset('source_pop_names', data=source_pop_names)
                            data_subgroup.create_dataset('syn_proportions', data=syn_proportions)
                    data_group = subgroup.create_group('pop_cell_positions')
                    for pop_name in context.pop_cell_positions:
                        data_subgroup = data_group.create_group(pop_name)
                        gids = np.array(list(context.pop_cell_positions[pop_name].keys()))
                        positions = np.array(list(context.pop_cell_positions[pop_name].values()))
                        data_subgroup.create_dataset('gids', data=gids, compression='gzip')
                        data_subgroup.create_dataset('positions', data=positions, compression='gzip')
                group = get_h5py_group(group, [context.run_trial], create=True)
                set_h5py_attr(group.attrs, 'trial_id', int(context.run_trial))
                set_h5py_attr(group.attrs, 'equilibrate', context.equilibrate)
                set_h5py_attr(group.attrs, 'voltages_exceed_threshold', voltages_exceed_threshold)
                subgroup = group.create_group('full_spike_times')
                for pop_name in full_spike_times_dict:
                    subgroup.create_group(pop_name)
                    for gid in full_spike_times_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=full_spike_times_dict[pop_name][gid], compression='gzip')
                subgroup = group.create_group('filter_results')
                data_group = subgroup.create_group('freq_tuning_index')
                for band in freq_tuning_index_dict:
                    data_group.create_group(band)
                    for pop_name in freq_tuning_index_dict[band]:
                        data_group[band].attrs[pop_name] = freq_tuning_index_dict[band][pop_name]
                data_group = subgroup.create_group('centroid_freq')
                for band in centroid_freq_dict:
                    data_group.create_group(band)
                    for pop_name in centroid_freq_dict[band]:
                        data_group[band].attrs[pop_name] = centroid_freq_dict[band][pop_name]
                data_group = subgroup.create_group('fft_f')
                for pop_name in fft_f_dict:
                    data_group.create_dataset(pop_name, data=fft_f_dict[pop_name], compression='gzip')
                data_group = subgroup.create_group('fft_power')
                for pop_name in fft_power_dict:
                    data_group.create_dataset(pop_name, data=fft_power_dict[pop_name], compression='gzip')
                data_group = subgroup.create_group('psd_f')
                for band in filter_psd_f_dict:
                    data_group.create_group(band)
                    for pop_name in filter_psd_f_dict[band]:
                        data_group[band].attrs[pop_name] = filter_psd_f_dict[band][pop_name]
                data_group = subgroup.create_group('psd_power')
                for band in filter_psd_power_dict:
                    data_group.create_group(band)
                    for pop_name in filter_psd_power_dict[band]:
                        data_group[band].attrs[pop_name] = filter_psd_power_dict[band][pop_name]
                subgroup = group.create_group('binned_firing_rates')
                for pop_name in binned_firing_rates_dict:
                    subgroup.create_group(pop_name)
                    for gid in binned_firing_rates_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=binned_firing_rates_dict[pop_name][gid], compression='gzip')
                subgroup = group.create_group('subset_full_voltage_recs')
                for pop_name in subset_full_voltage_rec_dict:
                    subgroup.create_group(pop_name)
                    for gid in subset_full_voltage_rec_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=subset_full_voltage_rec_dict[pop_name][gid], compression='gzip')
                group.create_dataset('full_rec_t', data=full_rec_t, compression='gzip')
                group.create_dataset('buffered_rec_t', data=buffered_rec_t, compression='gzip')
                group.create_dataset('rec_t', data=rec_t, compression='gzip')
            if context.debug and context.verbose > 0:
                print('optimize_simple_network: analyze_network_output_run: pid: %i; exporting data to file: %s took '
                      '%.2f s' % (os.getpid(), context.temp_output_path, time.time() - current_time))
                sys.stdout.flush()

        result = dict()

        result['E_mean_min_rate_run'] = mean_min_rate_dict['E']
        result['E_mean_peak_rate_run'] = mean_peak_rate_dict['E']
        result['I_mean_active_rate_run'] = np.mean(mean_rate_active_cells_dict['I'])
        result['FF_frac_active_run'] = np.mean(pop_fraction_active_dict['FF'])
        result['E_frac_active_run'] = np.mean(pop_fraction_active_dict['E'])
        result['I_frac_active_run'] = np.mean(pop_fraction_active_dict['I'])
        result['FF_theta_envelope_ratio_run'] = filter_envelope_ratio_dict['Theta']['FF']
        result['E_theta_envelope_ratio_run'] = filter_envelope_ratio_dict['Theta']['E']
        result['I_theta_envelope_ratio_run'] = filter_envelope_ratio_dict['Theta']['I']
        result['FF_gamma_envelope_ratio_run'] = filter_envelope_ratio_dict['Gamma']['FF']
        result['E_gamma_envelope_ratio_run'] = filter_envelope_ratio_dict['Gamma']['E']
        result['I_gamma_envelope_ratio_run'] = filter_envelope_ratio_dict['Gamma']['I']
        result['E_centroid_theta_freq_run'] = centroid_freq_dict['Theta']['E']
        result['I_centroid_theta_freq_run'] = centroid_freq_dict['Theta']['I']
        result['E_centroid_gamma_freq_run'] = centroid_freq_dict['Gamma']['E']
        result['I_centroid_gamma_freq_run'] = centroid_freq_dict['Gamma']['I']
        result['E_theta_tuning_index_run'] = freq_tuning_index_dict['Theta']['E']
        result['I_theta_tuning_index_run'] = freq_tuning_index_dict['Theta']['I']
        result['E_gamma_tuning_index_run'] = freq_tuning_index_dict['Gamma']['E']
        result['I_gamma_tuning_index_run'] = freq_tuning_index_dict['Gamma']['I']
        result['FF_ripple_envelope_ratio_run'] = filter_envelope_ratio_dict['Ripple']['FF']
        result['E_ripple_envelope_ratio_run'] = filter_envelope_ratio_dict['Ripple']['E']
        result['I_ripple_envelope_ratio_run'] = filter_envelope_ratio_dict['Ripple']['I']

        if any(voltages_exceed_threshold_list):
            if context.verbose > 0:
                print('optimize_simple_network: analyze_network_output_run: pid: %i; model failed - mean membrane '
                      'voltage in some Izhi cells exceeds spike threshold' % os.getpid())
                sys.stdout.flush()
            result['failed'] = True

        if context.interactive:
            context.update(locals())
        return result


def analyze_network_output_replay(network, ensemble_id=None, trial=None, model_id=None, export=False, plot=False):
    """

    :param network: :class:'SimpleNetwork'
    :param ensemble_id: int
    :param trial: int
    :param model_id: int
    :param export: bool
    :param plot: bool
    :return: dict
    """
    start_time = time.time()
    context.filter_bands = context.replay_filter_bands
    full_rec_t = np.arange(-context.buffer - context.equilibrate, context.duration + context.buffer + context.dt / 2.,
                           context.dt)
    buffered_rec_t = np.arange(-context.buffer, context.duration + context.buffer + context.dt / 2., context.dt)
    rec_t = np.arange(0., context.duration, context.dt)

    buffered_fine_binned_t_edges = \
        np.arange(-context.buffer, context.duration + context.buffer + context.fine_binned_dt / 2.,
                  context.fine_binned_dt)
    buffered_fine_binned_t = buffered_fine_binned_t_edges[:-1] + context.fine_binned_dt / 2.
    fine_binned_t_edges = np.arange(0., context.duration + context.fine_binned_dt / 2., context.fine_binned_dt)
    fine_binned_t = fine_binned_t_edges[:-1] + context.fine_binned_dt / 2.
    coarse_binned_t_edges = np.arange(0., context.duration + context.coarse_binned_dt / 2., context.coarse_binned_dt)
    coarse_binned_t = coarse_binned_t_edges[:-1] + context.coarse_binned_dt / 2.

    full_spike_times_dict = network.get_spike_times_dict()
    coarse_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict, coarse_binned_t_edges)
    firing_rates_from_coarse_binned_spike_count_dict = \
        get_firing_rate_from_binned_spike_count(coarse_binned_spike_count_dict, context.coarse_binned_dt)
    buffered_fine_binned_spike_count_dict = get_binned_spike_count_dict(full_spike_times_dict,
                                                                        buffered_fine_binned_t_edges)

    gathered_full_spike_times_dict_list = context.comm.gather(full_spike_times_dict, root=0)
    gathered_firing_rates_dict_list = context.comm.gather(firing_rates_from_coarse_binned_spike_count_dict, root=0)
    gathered_buffered_fine_binned_spike_count_dict_list = \
        context.comm.gather(buffered_fine_binned_spike_count_dict, root=0)

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
        print(
            'optimize_simple_network: analyze_network_output_replay: pid: %i; gathering data across ranks took %.2f s' %
            (os.getpid(), time.time() - start_time))
        sys.stdout.flush()
        current_time = time.time()

    if context.comm.rank == 0:
        full_spike_times_dict = merge_list_of_dict(gathered_full_spike_times_dict_list)
        firing_rates_dict = merge_list_of_dict(gathered_firing_rates_dict_list)
        buffered_fine_binned_spike_count_dict = merge_list_of_dict(gathered_buffered_fine_binned_spike_count_dict_list)

        if plot or export:
            subset_full_voltage_rec_dict = merge_list_of_dict(gathered_subset_full_voltage_rec_dict_list)
            connectivity_dict = merge_list_of_dict(connectivity_dict)
            connection_weights_dict = \
                merge_connection_weights_dicts(gathered_connection_target_gid_dict_list,
                                               gathered_connection_weights_dict_list)

        if context.debug and context.verbose > 0:
            print(
                'optimize_simple_network: analyze_network_output_replay: pid: %i; merging data structures for analysis '
                'took %.2f s' %
                (os.getpid(), time.time() - current_time))
            current_time = time.time()
            sys.stdout.flush()

        buffered_pop_mean_rate_from_binned_spike_count_dict = \
            get_pop_mean_rate_from_binned_spike_count(buffered_fine_binned_spike_count_dict, dt=context.fine_binned_dt)
        mean_min_rate_dict, mean_peak_rate_dict, mean_rate_active_cells_dict, pop_fraction_active_dict = \
            get_pop_activity_stats(firing_rates_dict, input_t=coarse_binned_t, threshold=context.active_rate_threshold,
                                   plot=plot)
        fft_f_dict, fft_power_dict, filter_psd_f_dict, filter_psd_power_dict, filter_envelope_dict, \
        filter_envelope_ratio_dict, centroid_freq_dict, freq_tuning_index_dict = \
            get_pop_bandpass_filtered_signal_stats(buffered_pop_mean_rate_from_binned_spike_count_dict,
                                                   context.filter_bands, input_t=buffered_fine_binned_t,
                                                   valid_t=fine_binned_t, output_t=fine_binned_t, pad=False,
                                                   plot=plot, verbose=context.verbose > 1)

        if context.debug and context.verbose > 0:
            print(
                'optimize_simple_network: analyze_network_output_replay: pid: %i; additional data analysis took %.2f '
                's' % (os.getpid(), time.time() - current_time))
            sys.stdout.flush()

        if plot:
            plot_firing_rate_heatmaps(firing_rates_dict, input_t=coarse_binned_t,
                                      tuning_peak_locs=context.tuning_peak_locs)
            plot_population_spike_rasters(buffered_fine_binned_spike_count_dict, input_t=buffered_fine_binned_t,
                                          tuning_peak_locs=context.tuning_peak_locs)

        if export:
            current_time = time.time()
            with h5py.File(context.temp_output_path, 'a') as f:
                run_group_key = 'simple_network_exported_replay_data'
                group = get_h5py_group(f, [model_id, run_group_key], create=True)
                shared_context_key = 'shared_context'
                if shared_context_key not in group:
                    subgroup = group.create_group(shared_context_key)
                    subgroup.create_dataset('param_names', data=np.array(context.param_names, dtype='S'),
                                            compression='gzip')
                    subgroup.create_dataset('x0', data=context.x0_array, compression='gzip')
                    set_h5py_attr(subgroup.attrs, 'network_id', context.network_id)
                    set_h5py_attr(subgroup.attrs, 'network_instance', context.network_instance)
                    set_h5py_attr(subgroup.attrs, 'replay_trial_offset', context.replay_trial_offset)
                    set_h5py_attr(subgroup.attrs, 'connectivity_type', context.connectivity_type)
                    set_h5py_attr(subgroup.attrs, 'duration', context.duration)
                    set_h5py_attr(subgroup.attrs, 'buffer', context.buffer)
                    set_h5py_attr(subgroup.attrs, 'stim_edge_duration', context.replay_stim_edge_duration)
                    set_h5py_attr(subgroup.attrs, 'active_rate_threshold', context.active_rate_threshold)
                    data_group = subgroup.create_group('pop_gid_ranges')
                    for pop_name in context.pop_gid_ranges:
                        data_group.create_dataset(pop_name, data=context.pop_gid_ranges[pop_name])
                    data_group = subgroup.create_group('filter_bands')
                    for filter, band in viewitems(context.filter_bands):
                        data_group.create_dataset(filter, data=band)
                    data_group = subgroup.create_group('connection_weights')
                    for target_pop_name in connection_weights_dict:
                        data_group.create_group(target_pop_name)
                        for source_pop_name in connection_weights_dict[target_pop_name]:
                            data_group[target_pop_name].create_dataset(
                                source_pop_name, data=connection_weights_dict[target_pop_name][source_pop_name],
                                compression='gzip')
                    if len(context.tuning_peak_locs) > 0:
                        data_group = subgroup.create_group('tuning_peak_locs')
                        for pop_name in context.tuning_peak_locs:
                            if len(context.tuning_peak_locs[pop_name]) > 0:
                                data_group.create_group(pop_name)
                                target_gids = np.array(list(context.tuning_peak_locs[pop_name].keys()))
                                peak_locs = np.array(list(context.tuning_peak_locs[pop_name].values()))
                                data_group[pop_name].create_dataset('target_gids', data=target_gids, compression='gzip')
                                data_group[pop_name].create_dataset('peak_locs', data=peak_locs, compression='gzip')
                    data_group = subgroup.create_group('connectivity')
                    for target_pop_name in connectivity_dict:
                        data_group.create_group(target_pop_name)
                        for target_gid in connectivity_dict[target_pop_name]:
                            data_subgroup = data_group[target_pop_name].create_group(str(target_gid))
                            for source_pop_name in connectivity_dict[target_pop_name][target_gid]:
                                data_subgroup.create_dataset(source_pop_name,
                                                             data=connectivity_dict[target_pop_name][target_gid][
                                                                 source_pop_name])
                    data_group = subgroup.create_group('pop_syn_proportions')
                    for target_pop_name in context.pop_syn_proportions:
                        data_group.create_group(target_pop_name)
                        for syn_type in context.pop_syn_proportions[target_pop_name]:
                            data_subgroup = data_group[target_pop_name].create_group(syn_type)
                            source_pop_names = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].keys()), dtype='S')
                            syn_proportions = \
                                np.array(list(context.pop_syn_proportions[target_pop_name][syn_type].values()))
                            data_subgroup.create_dataset('source_pop_names', data=source_pop_names)
                            data_subgroup.create_dataset('syn_proportions', data=syn_proportions)
                    data_group = subgroup.create_group('pop_cell_positions')
                    for pop_name in context.pop_cell_positions:
                        data_subgroup = data_group.create_group(pop_name)
                        gids = np.array(list(context.pop_cell_positions[pop_name].keys()))
                        positions = np.array(list(context.pop_cell_positions[pop_name].values()))
                        data_subgroup.create_dataset('gids', data=gids, compression='gzip')
                        data_subgroup.create_dataset('positions', data=positions, compression='gzip')
                group = get_h5py_group(group, [trial], create=True)
                set_h5py_attr(group.attrs, 'ensemble_id', int(ensemble_id))
                set_h5py_attr(group.attrs, 'trial_id', int(trial))
                set_h5py_attr(group.attrs, 'equilibrate', context.equilibrate)
                set_h5py_attr(group.attrs, 'voltages_exceed_threshold', voltages_exceed_threshold)
                subgroup = group.create_group('full_spike_times')
                for pop_name in full_spike_times_dict:
                    subgroup.create_group(pop_name)
                    for gid in full_spike_times_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=full_spike_times_dict[pop_name][gid], compression='gzip')
                subgroup = group.create_group('filter_results')
                data_group = subgroup.create_group('freq_tuning_index')
                for band in freq_tuning_index_dict:
                    data_group.create_group(band)
                    for pop_name in freq_tuning_index_dict[band]:
                        data_group[band].attrs[pop_name] = freq_tuning_index_dict[band][pop_name]
                data_group = subgroup.create_group('centroid_freq')
                for band in centroid_freq_dict:
                    data_group.create_group(band)
                    for pop_name in centroid_freq_dict[band]:
                        data_group[band].attrs[pop_name] = centroid_freq_dict[band][pop_name]
                data_group = subgroup.create_group('fft_f')
                for pop_name in fft_f_dict:
                    data_group.create_dataset(pop_name, data=fft_f_dict[pop_name], compression='gzip')
                data_group = subgroup.create_group('fft_power')
                for pop_name in fft_power_dict:
                    data_group.create_dataset(pop_name, data=fft_power_dict[pop_name], compression='gzip')
                data_group = subgroup.create_group('psd_f')
                for band in filter_psd_f_dict:
                    data_group.create_group(band)
                    for pop_name in filter_psd_f_dict[band]:
                        data_group[band].attrs[pop_name] = filter_psd_f_dict[band][pop_name]
                data_group = subgroup.create_group('psd_power')
                for band in filter_psd_power_dict:
                    data_group.create_group(band)
                    for pop_name in filter_psd_power_dict[band]:
                        data_group[band].attrs[pop_name] = filter_psd_power_dict[band][pop_name]
                subgroup = group.create_group('subset_full_voltage_recs')
                for pop_name in subset_full_voltage_rec_dict:
                    subgroup.create_group(pop_name)
                    for gid in subset_full_voltage_rec_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=subset_full_voltage_rec_dict[pop_name][gid], compression='gzip')
                group.create_dataset('full_rec_t', data=full_rec_t, compression='gzip')
                group.create_dataset('buffered_rec_t', data=buffered_rec_t, compression='gzip')
                group.create_dataset('rec_t', data=rec_t, compression='gzip')
            if context.verbose > 0:
                print('optimize_simple_network_replay: analyze_network_output_replay: pid: %i; trial: %i; exporting '
                      'data to file: %s took %.2f s' % (os.getpid(), trial, context.temp_output_path,
                                                        time.time() - current_time))
                sys.stdout.flush()

        FF_mean_frac_active_replay = np.mean(pop_fraction_active_dict['FF'])
        E_frac_active_replay_residuals = \
            np.sum(np.abs(np.subtract(pop_fraction_active_dict['E'], FF_mean_frac_active_replay)))

        result = dict()

        result['FF_frac_active_replay'] = FF_mean_frac_active_replay
        result['E_frac_active_replay'] = np.mean(pop_fraction_active_dict['E'])
        result['I_frac_active_replay'] = np.mean(pop_fraction_active_dict['I'])
        result['FF_mean_active_rate_replay'] = np.mean(mean_rate_active_cells_dict['FF'])
        result['E_mean_active_rate_replay'] = np.mean(mean_rate_active_cells_dict['E'])
        result['I_mean_active_rate_replay'] = np.mean(mean_rate_active_cells_dict['I'])
        result['FF_ripple_envelope_ratio_replay'] = filter_envelope_ratio_dict['Ripple']['FF']
        result['E_ripple_envelope_ratio_replay'] = filter_envelope_ratio_dict['Ripple']['E']
        result['I_ripple_envelope_ratio_replay'] = filter_envelope_ratio_dict['Ripple']['I']
        result['E_frac_active_replay_residuals'] = E_frac_active_replay_residuals

        if any(voltages_exceed_threshold_list):
            if context.verbose > 0:
                print('optimize_simple_network: analyze_network_output_replay: pid: %i; model failed - mean membrane '
                      'voltage in some Izhi cells exceeds spike threshold' % os.getpid())
                sys.stdout.flush()
            result['failed'] = True

        if context.interactive:
            context.update(locals())
        return result


def filter_features_replay(primitives, features, model_id=None, export=False):
    """

    :param primitives: list of dict
    :param features: dict
    :param model_id: int
    :param export: bool
    :return: dict
    """
    feature_lists = defaultdict(list)
    for result in primitives:
        for key, val in result.items():
            feature_lists[key].append(val)
    new_features = {}
    for key in feature_lists:
        new_features[key] = np.mean(feature_lists[key])

    return new_features


def compute_features_run(x, model_id=None, export=False):
    """

    :param x: array
    :param model_id: int
    :param export: bool
    :return: dict
    """
    start_time = time.time()
    update_source_contexts(x, context)
    spikes_seed = context.base_seed + [2, context.run_stim_type_seed, context.run_trial]
    context.buffer = context.run_buffer
    context.duration = context.run_duration
    context.tstop = context.run_tstop
    context.baks_pad_dur = context.run_baks_pad_dur
    context.baks_wrap_around = context.run_baks_wrap_around

    context.pc.gid_clear()
    context.network = SimpleNetwork(
        pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
        pop_cell_types=context.pop_cell_types, pop_syn_counts=context.pop_syn_counts,
        pop_syn_proportions=context.pop_syn_proportions, connection_weights_mean=context.connection_weights_mean,
        connection_weights_norm_sigma=context.connection_weights_norm_sigma,
        syn_mech_params=context.syn_mech_params, tstop=context.tstop, duration=context.duration, buffer=context.buffer,
        equilibrate=context.equilibrate, dt=context.dt, delay=context.delay, verbose=context.verbose,
        debug=context.debug)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; network initialization (run) took %.2f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    context.network.set_input_pattern(context.input_types, input_mean_rates=context.input_mean_rates_run,
                                      input_min_rates=context.input_min_rates_run,
                                      input_max_rates=context.input_max_rates_run,
                                      input_norm_tuning_widths=context.input_norm_tuning_widths,
                                      tuning_peak_locs=context.tuning_peak_locs,
                                      track_wrap_around=context.track_wrap_around, spikes_seed=spikes_seed,
                                      tuning_duration=context.tuning_duration)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; setting network input pattern (run) took %.2f s' %
              (os.getpid(), time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    if context.connectivity_type == 'uniform':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed)
    elif context.connectivity_type == 'gaussian':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed,
                                      pop_axon_extents=context.pop_axon_extents,
                                      pop_cell_positions=context.pop_cell_positions)

    context.network.assign_connection_weights(
        default_weight_distribution_type=context.default_weight_distribution_type,
        connection_weight_distribution_types=context.connection_weight_distribution_types,
        weights_seed=context.weights_seed)

    if context.structured_weights:
        context.network.structure_connection_weights(structured_weight_params=context.structured_weight_params,
                                                     tuning_peak_locs=context.tuning_peak_locs,
                                                     wrap_around=context.track_wrap_around,
                                                     tuning_duration=context.tuning_duration)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; building network connections (run) took %.2f s' %
              (os.getpid(), time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()
    """
    if context.debug:
        context.update(locals())
        return dict()
    """

    context.network.run()
    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; model_id: %i; network simulation (run) took %.2f s' %
              (os.getpid(), model_id, time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    results = analyze_network_output_run(context.network, model_id=model_id, export=export, plot=context.plot)
    if context.comm.rank == 0:
        if context.verbose > 0:
            print('optimize_simple_network: pid: %i; model_id: %i; analysis of network simulation results (run) '
                  'took %.2f s' % (os.getpid(), model_id, time.time() - current_time))
            sys.stdout.flush()
        if results is None:
            return dict()
        return results


def compute_features_replay(x, ensemble_id=None, trial=None, model_id=None, export=False):
    """

    :param x: array
    :param ensemble_id: int
    :param trial: int
    :param model_id: int
    :param export: bool
    :return: dict
    """
    start_time = time.time()
    update_source_contexts(x, context)

    trial_selection_seed = context.selection_seed + [int(ensemble_id)]
    trial_spikes_seed = context.base_seed + [2, context.replay_stim_type_seed, int(ensemble_id), int(trial)]
    context.buffer = context.replay_buffer
    context.duration = context.replay_duration
    context.tstop = context.replay_tstop
    context.baks_pad_dur = context.replay_baks_pad_dur
    context.baks_wrap_around = context.replay_baks_wrap_around

    context.pc.gid_clear()
    context.network = SimpleNetwork(
        pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
        pop_cell_types=context.pop_cell_types, pop_syn_counts=context.pop_syn_counts,
        pop_syn_proportions=context.pop_syn_proportions, connection_weights_mean=context.connection_weights_mean,
        connection_weights_norm_sigma=context.connection_weights_norm_sigma,
        syn_mech_params=context.syn_mech_params, tstop=context.tstop, duration=context.duration, buffer=context.buffer,
        equilibrate=context.equilibrate, dt=context.dt, delay=context.delay, verbose=context.verbose,
        debug=context.debug)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; network initialization (replay) took %.2f s' %
              (os.getpid(), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    context.network.set_offline_input_pattern(
        context.input_types, input_offline_min_rates=context.input_offline_min_rates,
        input_offline_mean_rates=context.input_offline_mean_rates,
        input_offline_fraction_active=context.input_offline_fraction_active, tuning_peak_locs=context.tuning_peak_locs,
        track_wrap_around=context.track_wrap_around, stim_edge_duration=context.replay_stim_edge_duration,
        selection_seed=trial_selection_seed, spikes_seed=trial_spikes_seed, tuning_duration=context.tuning_duration)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; setting network input pattern (replay) took %.2f s' %
              (os.getpid(), time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    if context.connectivity_type == 'uniform':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed)
    elif context.connectivity_type == 'gaussian':
        context.network.connect_cells(connectivity_type=context.connectivity_type,
                                      connection_seed=context.connection_seed,
                                      pop_axon_extents=context.pop_axon_extents,
                                      pop_cell_positions=context.pop_cell_positions)

    context.network.assign_connection_weights(
        default_weight_distribution_type=context.default_weight_distribution_type,
        connection_weight_distribution_types=context.connection_weight_distribution_types,
        weights_seed=context.weights_seed)

    if context.structured_weights:
        context.network.structure_connection_weights(structured_weight_params=context.structured_weight_params,
                                                     tuning_peak_locs=context.tuning_peak_locs,
                                                     wrap_around=context.track_wrap_around,
                                                     tuning_duration=context.tuning_duration)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; building network connections (replay) took %.2f s' %
              (os.getpid(), time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()
    """
    if context.debug:
        context.update(locals())
        return dict()
    """

    context.network.run()
    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; model_id: %i; trial: %i; network simulation (replay) took %.2f s' %
              (os.getpid(), model_id, trial, time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    results = analyze_network_output_replay(context.network, ensemble_id=ensemble_id, trial=trial, model_id=model_id,
                                            export=export, plot=context.plot)
    if context.comm.rank == 0:
        if context.verbose > 0:
            print('optimize_simple_network: pid: %i; model_id: %i; trial: %i; analysis of network simulation results '
                  '(replay) took %.2f s' % (os.getpid(), model_id, trial, time.time() - current_time))
            sys.stdout.flush()
        if results is None:
            return dict()
        return results


def get_objectives(features, model_id=None, export=False):
    """

    :param features: dict
    :param model_id: int
    :param export: bool
    :return: tuple of dict
    """
    if context.comm.rank == 0:
        objectives = {}
        for objective_name in context.objective_names:
            if (objective_name.find('tuning_index') != -1 and
                    features[objective_name] >= context.target_val[objective_name]):
                objectives[objective_name] = 0.
            elif objective_name.find('ripple_envelope_score') != -1:
                base_name = 'ripple_envelope_ratio'
                if objective_name == 'E_ripple_envelope_score':
                    pop_name = 'E'
                elif objective_name == 'I_ripple_envelope_score':
                    pop_name = 'I'
                run_val = features['%s_%s_run' % (pop_name, base_name)]
                replay_val = features['%s_%s_replay' % (pop_name, base_name)]
                if replay_val > run_val:
                    objectives[objective_name] = 0.
                else:
                    objectives[objective_name] = ((replay_val - run_val) /
                                                  context.target_range[objective_name]) ** 2.
            elif objective_name == 'E_mean_active_rate_replay':
                E_val = features['E_mean_active_rate_replay']
                FF_val = features['FF_mean_active_rate_replay']
                if E_val < FF_val:
                    objectives[objective_name] = 0.
                else:
                    objectives[objective_name] = ((E_val - FF_val) /
                                                  context.target_range[objective_name]) ** 2.
            else:
                objectives[objective_name] = ((context.target_val[objective_name] - features[objective_name]) /
                                              context.target_range[objective_name]) ** 2.
        return features, objectives


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
