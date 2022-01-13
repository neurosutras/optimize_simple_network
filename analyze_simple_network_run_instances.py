import click
from nested.parallel import get_parallel_interface
from nested.utils import get_unknown_click_arg_dict
from nested.optimize_utils import config_parallel_interface, Context
from simple_network_analysis_utils import *

context = Context()


def get_args_analyze_instances(data_file_name_list, data_dir, data_key, example_trial, example_instance,
                               fine_binned_dt, coarse_binned_dt, filter_order, filter_label_dict, filter_color_dict,
                               filter_xlim_dict, pop_order, label_dict, color_dict, plot, verbose):
    """

    :param data_file_name_list: list of str
    :param data_dir: str
    :param data_key: str
    :param example_trial: int
    :param example_instance: int
    :param fine_binned_dt: float
    :param coarse_binned_dt: float
    :param filter_order: list of str
    :param filter_label_dict: dict
    :param filter_color_dict: dict
    :param filter_xlim_dict: dict of tuple of float
    :param pop_order: list
    :param label_dict: dict
    :param color_dict: dict
    :param plot: bool
    :param verbose: bool
    :return: list of list of args
    """
    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'
    example_trial_key = str(example_trial)
    network_instance_found = False

    sequences = []
    for data_file_name in data_file_name_list:
        args = []
        data_file_path = data_dir + '/' + data_file_name
        args.append(data_file_path)
        args.append(data_key)
        with h5py.File(data_file_path, 'r') as f:
            group = get_h5py_group(f, [data_key, group_key])
            subgroup = group[shared_context_key]
            network_instance = get_h5py_attr(subgroup.attrs, 'network_instance')

            trial_keys = [key for key in group if key != shared_context_key]
            if example_instance is not None and example_instance == network_instance:
                network_instance_found = True
                if example_trial is not None:
                    if example_trial_key in trial_keys:
                        args.append(example_trial)
                    else:
                        raise RuntimeError('analyze_simple_network_run_instances: example_trial: %i not found in '
                                           'file: %s with network_instance: %i' %
                                           (example_trial, data_file_path, network_instance))
                else:
                    args.append(None)
                this_plot = plot
            else:
                this_plot = False
                args.append(None)
        args.extend([fine_binned_dt, coarse_binned_dt, filter_order, filter_label_dict, filter_color_dict,
                     filter_xlim_dict, pop_order, label_dict, color_dict, this_plot, verbose])
        sequences.append(args)

    if example_instance is not None and not network_instance_found:
        raise RuntimeError('analyze_simple_network_run_instances: example_instance: %i not found' % (example_instance))

    return list(zip(*sequences))


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              default='data')
@click.option("--export-data-file-path", type=click.Path(exists=False, file_okay=True, dir_okay=False), default=None)
@click.option("--example-trial", type=int, default=None)
@click.option("--example-instance", type=int, default=None)
@click.option("--export-data-key", type=str, default='0')
@click.option("--fine-binned-dt", type=float, default=1.)
@click.option("--coarse-binned-dt", type=float, default=20.)
@click.option("--model-key", type=str, default='J')
@click.option("--export", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, config_file_path, data_dir, export_data_file_path, example_trial, example_instance, export_data_key,
         fine_binned_dt, coarse_binned_dt, model_key, export, interactive, verbose, plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_file_path: str (path to .hdf5 file to export instance summary data)
    :param example_trial: int; plot individual trial data from this trial
    :param example_instance: plot trial averaged data from this network instance
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param fine_binned_dt: float (ms)
    :param coarse_binned_dt: float (ms)
    :param model_key: str (identifier for exported instance summary data)
    :param export: bool
    :param interactive: bool
    :param verbose: bool
    :param plot: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())
    kwargs = get_unknown_click_arg_dict(cli.args)

    context.interface = get_parallel_interface(source_file=__file__, source_package=__package__, **kwargs)
    context.interface.start(disp=verbose)
    context.interface.ensure_controller()

    config_parallel_interface(__file__, config_file_path=config_file_path, interface=context.interface)

    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None
    if 'filter_order' not in context():
        context.filter_order = None
    if 'filter_label_dict' not in context():
        context.filter_label_dict = None
    if 'filter_color_dict' not in context():
        context.filter_color_dict = None
        filter_xlim_dict
    if 'filter_xlim_dict' not in context():
        context.filter_xlim_dict = None

    current_time = time.time()

    sequences = get_args_analyze_instances(context.data_file_name_list, data_dir, export_data_key, example_trial,
                                           example_instance, fine_binned_dt, coarse_binned_dt, context.filter_order,
                                           context.filter_label_dict, context.filter_color_dict,
                                           context.filter_xlim_dict, context.pop_order, context.label_dict,
                                           context.color_dict, plot, verbose)

    results = context.interface.map(analyze_simple_network_run_data_from_file, *sequences)

    binned_t_edges_list, sorted_trial_averaged_firing_rate_matrix_dict_list, sorted_gid_dict_list, \
    centered_firing_rate_mean_dict_list, mean_rate_active_cells_mean_dict_list, pop_fraction_active_mean_dict_list, \
    fft_f_list, fft_power_mean_dict_list, fft_f_nested_gamma_list, fft_power_nested_gamma_mean_dict_list,\
        modulation_depth_dict_list, delta_peak_locs_dict_list = zip(*results)

    binned_t_edges = binned_t_edges_list[0]
    centered_firing_rate_mean_dict, centered_firing_rate_sem_dict = \
        analyze_selectivity_across_instances(centered_firing_rate_mean_dict_list)

    mean_rate_active_cells_mean_dict, mean_rate_active_cells_sem_dict, pop_fraction_active_mean_dict, \
    pop_fraction_active_sem_dict = \
        get_trial_averaged_pop_activity_stats(mean_rate_active_cells_mean_dict_list,
                                              pop_fraction_active_mean_dict_list)

    fft_f = fft_f_list[0]
    fft_power_mean_dict, fft_power_sem_dict = get_trial_averaged_fft_power(fft_f, fft_power_mean_dict_list)

    fft_f_nested_gamma = fft_f_nested_gamma_list[0]
    if fft_f_nested_gamma is not None:
        fft_power_nested_gamma_mean_dict, fft_power_nested_gamma_sem_dict = \
            get_trial_averaged_fft_power(fft_f_nested_gamma, fft_power_nested_gamma_mean_dict_list)

    modulation_depth_instances_dict, delta_peak_locs_instances_dict = \
        analyze_spatial_modulation_across_instances(modulation_depth_dict_list, delta_peak_locs_dict_list)

    print('analyze_simple_network_run_instances took %.1f s to analyze data from %i network instances from file' %
          (time.time() - current_time, len(context.data_file_name_list)))
    sys.stdout.flush()
    current_time = time.time()

    if export:
        if export_data_file_path is None:
            raise IOError('analyze_simple_network_run_instances: invalid export_data_file_path: %s' %
                          export_data_file_path)
        with h5py.File(export_data_file_path, 'a') as f:
            group = get_h5py_group(f, ['shared_context'], create=True)
            if 'fft_f' not in group:
                group.create_dataset('fft_f', data=fft_f)
            if 'fft_f_nested_gamma' not in group:
                group.create_dataset('fft_f_nested_gamma', data=fft_f_nested_gamma)
            group = get_h5py_group(f, [model_key], create=True)

            subgroup = group.create_group('fft_power')
            fft_power_mean_list_dict = {}
            for fft_power_mean_dict in fft_power_mean_dict_list:
                for pop_name in fft_power_mean_dict:
                    if pop_name not in fft_power_mean_list_dict:
                        fft_power_mean_list_dict[pop_name] = []
                    fft_power_mean_list_dict[pop_name].append(fft_power_mean_dict[pop_name])
            for pop_name in fft_power_mean_list_dict:
                subgroup.create_dataset(pop_name, data=np.array(fft_power_mean_list_dict[pop_name]))

            subgroup = group.create_group('fft_power_nested_gamma')
            fft_power_nested_gamma_mean_list_dict = {}
            for fft_power_nested_gamma_mean_dict in fft_power_nested_gamma_mean_dict_list:
                for pop_name in fft_power_nested_gamma_mean_dict:
                    if pop_name not in fft_power_nested_gamma_mean_list_dict:
                        fft_power_nested_gamma_mean_list_dict[pop_name] = []
                    fft_power_nested_gamma_mean_list_dict[pop_name].append(fft_power_nested_gamma_mean_dict[pop_name])
            for pop_name in fft_power_nested_gamma_mean_list_dict:
                subgroup.create_dataset(pop_name, data=np.array(fft_power_nested_gamma_mean_list_dict[pop_name]))

            subgroup = group.create_group('spatial_modulation_depth')
            for condition in modulation_depth_instances_dict:
                subgroup.create_group(condition)
                for pop_name in modulation_depth_instances_dict[condition]:
                    subgroup[condition].create_dataset(pop_name,
                                                       data=modulation_depth_instances_dict[condition][pop_name])

            subgroup = group.create_group('delta_peak_locs')
            for pop_name in delta_peak_locs_instances_dict:
                subgroup.create_dataset(pop_name, data=delta_peak_locs_instances_dict[pop_name])
        print('analyze_simple_network_run_instances took %.1f s to export data from %i network instances (model: %s) to '
              'file: %s' % (time.time() - current_time, len(context.data_file_name_list), model_key,
                            export_data_file_path))
        sys.stdout.flush()

    if plot:
        context.interface.apply(plt.show)
        plot_average_selectivity(binned_t_edges, centered_firing_rate_mean_dict, centered_firing_rate_sem_dict,
                                 pop_order=context.pop_order, label_dict=context.label_dict,
                                 color_dict=context.color_dict)
        plot_selectivity_input_output(modulation_depth_instances_dict, delta_peak_locs_instances_dict,
                                      pop_order=['E', 'I'], label_dict=context.label_dict,
                                      color_dict=context.color_dict)
        plot_pop_activity_stats(binned_t_edges, mean_rate_active_cells_mean_dict, pop_fraction_active_mean_dict, \
                                mean_rate_active_cells_sem_dict, pop_fraction_active_sem_dict,
                                pop_order=context.pop_order, label_dict=context.label_dict,
                                color_dict=context.color_dict)
        plot_rhythmicity_psd(fft_f, fft_power_mean_dict, fft_power_sem_dict, pop_order=context.pop_order,
                         label_dict=context.label_dict, color_dict=context.color_dict)
        if fft_f_nested_gamma is not None:
            plot_rhythmicity_psd(fft_f_nested_gamma, fft_power_nested_gamma_mean_dict, fft_power_sem_dict,
                                 pop_order=context.pop_order, label_dict=context.label_dict,
                                 color_dict=context.color_dict)
        plt.show()

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
