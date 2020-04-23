from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
import click

context = Context()


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/simulate_simple_network_replay_J_uniform_c_structured_w_gaussian_inp_config.yaml')
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--export-file-path", type=str, default=None)
@click.option("--label", type=str, default=None)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--simulate", type=bool, default=True)
@click.option("--merge_output_files", is_flag=True)
@click.pass_context
def main(cli, config_file_path, export, output_dir, export_file_path, label, interactive, verbose, plot, debug,
         simulate, merge_output_files):
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
    :param merge_output_files: bool
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

    config_parallel_interface(__file__, config_file_path=config_file_path, output_dir=output_dir, export=export,
                              export_file_path=export_file_path, label=label, disp=context.disp,
                              interface=context.interface, verbose=verbose, plot=plot, debug=debug, **kwargs)

    if simulate:
        args = context.interface.execute(get_args_static_offline_stimulus)
        sequences = args + [list(range(context.num_input_patterns))] + [[context.export] * context.num_input_patterns]
        context.interface.map(simulate_network_replay, *sequences)

        if context.export:
            if merge_output_files:
                merge_exported_data(context, export_file_path=context.export_file_path,
                                    output_dir=context.output_dir, verbose=context.verbose > 0)
            else:
                write_merge_path_list_to_yaml(context, export_file_path=context.export_file_path,
                                              output_dir=context.output_dir, verbose=context.verbose > 0)
        sys.stdout.flush()
        time.sleep(.1)

    if context.plot:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


def get_args_static_offline_stimulus():

    return [[context.spikes_seed + 1000000 * i for i in range(context.num_input_patterns)]]


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
    
    if 'input_mean_rates' not in context():
        input_mean_rates = None
    if 'input_min_rates' not in context():
        input_min_rates = None
    if 'input_max_rates' not in context():
        input_max_rates = None
    if 'input_norm_tuning_widths' not in context():
        input_norm_tuning_widths = None
    context.num_input_patterns = int(context.num_input_patterns)
    
    connection_weights_mean = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    connection_weights_norm_sigma = defaultdict(dict)  # {'target_pop_name': {'source_pop_name': float} }
    if 'connection_weight_distribution_types' not in context() or context.connection_weight_distribution_types is None:
        context.connection_weight_distribution_types = dict()
    if 'structured_weight_params' not in context() or context.structured_weight_params is None:
        context.structured_weight_params = dict()
        structured_weights = False
    else:
        structured_weights = True
    local_np_random = np.random.RandomState()
    if any([this_input_type == 'gaussian' for this_input_type in context.input_types.values()]) or structured_weights:
        if 'tuning_seed' not in context() or context.tuning_seed is None:
            raise RuntimeError('optimize_simple_network: config_file missing required parameter: tuning_seed')
        local_np_random.seed(int(context.tuning_seed))

    syn_mech_params = defaultdict(lambda: defaultdict(dict))
    if 'default_syn_mech_params' in context() and context.default_syn_mech_params is not None:
        for target_pop_name in context.default_syn_mech_params:
            for source_pop_name in context.default_syn_mech_params[target_pop_name]:
                for param_name in context.default_syn_mech_params[target_pop_name][source_pop_name]:
                    syn_mech_params[target_pop_name][source_pop_name][param_name] = \
                        context.default_syn_mech_params[target_pop_name][source_pop_name][param_name]

    delay = 1.  # ms
    equilibrate = 250.  # ms
    buffer = 100.  # ms
    duration = 250.  # ms
    stim_edge_duration = (150., 25.)  # ms
    tuning_duration = 3000.  # ms
    tstop = int(equilibrate + duration + 2. * buffer)  # ms
    dt = 0.025
    binned_dt = 1.  # ms
    filter_dt = 1.  # ms
    active_rate_threshold = 1.  # Hz
    baks_alpha = 4.7725100028345535
    baks_beta = 0.41969058927343522
    baks_pad_dur = 0.  # ms
    baks_wrap_around = False
    track_wrap_around = True
    filter_bands = {'Theta': [4., 10.], 'Gamma': [30., 100.], 'Ripple': [30., 150.]}
    max_num_cells_export_voltage_rec = 500

    if context.comm.rank == 0:
        tuning_peak_locs = dict()  # {'pop_name': {'gid': float} }

        for pop_name in context.input_types:
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

        for target_pop_name in context.structured_weight_params:
            if target_pop_name not in tuning_peak_locs:
                tuning_peak_locs[target_pop_name] = dict()
            this_norm_tuning_width = \
                context.structured_weight_params[target_pop_name]['norm_tuning_width']
            this_tuning_width = tuning_duration * this_norm_tuning_width
            this_sigma = this_tuning_width / 3. / np.sqrt(2.)
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
        for param_name in ['spatial_dim', 'location_seed']:
            if param_name not in context():
                raise RuntimeError('optimize_simple_network: parameter required for gaussian connectivity not found: '
                                   '%s' % param_name)

        if context.comm.rank == 0:
            pop_cell_positions = dict()
            for pop_name in pop_gid_ranges:
                for gid in range(pop_gid_ranges[pop_name][0], pop_gid_ranges[pop_name][1]):
                    local_np_random.seed(int(context.location_seed) + gid)
                    if pop_name not in pop_cell_positions:
                        pop_cell_positions[pop_name] = dict()
                    pop_cell_positions[pop_name][gid] = local_np_random.uniform(-1., 1., size=context.spatial_dim)
        else:
            pop_cell_positions = None
        pop_cell_positions = context.comm.bcast(pop_cell_positions, root=0)
    else:
        pop_cell_positions = dict()

    context.update(locals())

    context.x0_array = param_dict_to_array(context.x0, context.param_names)
    update_context(context.x0_array)

    """
    if context.debug:
        context.update(locals())
        return
    """

    context.network = SimpleNetwork(
        pc=context.pc, pop_sizes=context.pop_sizes, pop_gid_ranges=context.pop_gid_ranges,
        pop_cell_types=context.pop_cell_types, pop_syn_counts=context.pop_syn_counts,
        pop_syn_proportions=context.pop_syn_proportions, connection_weights_mean=context.connection_weights_mean,
        connection_weights_norm_sigma=context.connection_weights_norm_sigma,
        syn_mech_params=context.syn_mech_params, tstop=context.tstop, duration=context.duration, buffer=context.buffer,
        equilibrate=context.equilibrate, dt=context.dt, delay=context.delay, verbose=context.verbose,
        debug=context.debug)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; network initialization took %.2f s' %
              (os.getpid(), time.time() - start_time))
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
        print('optimize_simple_network: pid: %i; building network connections took %.2f s' %
              (os.getpid(), time.time() - current_time))
        sys.stdout.flush()


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
                subgroup = group.create_group('buffered_firing_rates')
                for pop_name in buffered_firing_rates_dict:
                    subgroup.create_group(pop_name)
                    for gid in buffered_firing_rates_dict[pop_name]:
                        subgroup[pop_name].create_dataset(
                            str(gid), data=buffered_firing_rates_dict[pop_name][gid], compression='gzip')
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


def simulate_network_replay(spikes_seed, model_id=None, export=False):
    """

    :param spikes_seed: int
    :param model_id: int
    :param export: bool
    """
    start_time = time.time()

    context.network.set_offline_input_pattern(
        context.input_types, input_offline_min_rates=context.input_offline_min_rates,
        input_offline_mean_rates=context.input_offline_mean_rates,
        input_offline_fraction_active=context.input_offline_fraction_active, tuning_peak_locs=context.tuning_peak_locs,
        track_wrap_around=context.track_wrap_around,stim_edge_duration=context.stim_edge_duration,
        spikes_seed=int(spikes_seed), tuning_duration=context.tuning_duration)

    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; model_id: %i; setting network input pattern took %.2f s' %
              (os.getpid(), model_id, time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    """
    if context.debug:
        context.update(locals())
        return dict()
    """

    context.network.run()
    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; model_id: %i; network simulation took %.2f s' %
              (os.getpid(), model_id, time.time() - current_time))
        sys.stdout.flush()
    current_time = time.time()

    analyze_network_output(context.network, model_id=model_id, export=export, plot=context.plot)
    if context.comm.rank == 0 and context.verbose > 0:
        print('optimize_simple_network: pid: %i; model_id: %i; analysis of network simulation results took %.2f s' %
              (os.getpid(), model_id, time.time() - current_time))
        sys.stdout.flush()


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
