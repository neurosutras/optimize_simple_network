import decode_simple_network_replay
import click
from nested.parallel import *
from nested.optimize_utils import *
from simple_network_utils import *
from decode_simple_network_replay import *


context = decode_simple_network_replay.context


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True, ))
@click.option("--template-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--decode-data-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--template-group-key", type=str, default='simple_network_exported_run_data')
@click.option("--decode-group-key", type=str, default='simple_network_exported_run_data')
@click.option("--export-data-key", type=int, default=0)
@click.option("--plot-n-trials", type=int, default=10)
@click.option("--template-window-dur", type=float, default=20.)
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--step-dur", type=float, default=20.)
@click.option("--output-dir", type=str, default='data')
@click.option("--interactive", is_flag=True)
@click.option("--verbose", type=int, default=2)
@click.option("--export", is_flag=True)
@click.option("--plot", is_flag=True)
@click.option("--debug", is_flag=True)
@click.pass_context
def main(cli, template_data_file_path, decode_data_file_path, template_group_key, decode_group_key, export_data_key,
         plot_n_trials, template_window_dur, decode_window_dur, step_dur, output_dir, interactive, verbose, export,
         plot, debug):
    """

    :param cli: contains unrecognized args as list of str
    :param template_data_file_path: str (path)
    :param decode_data_file_path: str (path)
    :param template_group_key: str
    :param decode_group_key: str
    :param export_data_key: str
    :param plot_n_trials: int
    :param template_window_dur: float
    :param decode_window_dur: float
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
                              template_group_key=template_group_key, decode_group_key=decode_group_key, **kwargs)

    context.interface.update_worker_contexts(template_duration=context.template_duration,
                                             template_window_dur=context.template_window_dur,
                                             decode_duration=context.decode_duration,
                                             decode_window_dur=context.decode_window_dur)

    start_time = time.time()

    if not context.template_data_is_processed:
        context.sorted_gid_dict, context.template_firing_rate_matrix_dict = \
            process_template_data(context.template_data_file_path, context.template_trial_keys,
                                  context.template_group_key, context.export_data_key, context.template_window_dur,
                                  export=context.export, disp=context.disp)
    else:
        context.sorted_gid_dict, context.template_firing_rate_matrix_dict = \
            load_processed_template_data(context.template_data_file_path, context.export_data_key)

    if context.plot:
        plot_firing_rate_heatmaps_from_matrix(context.template_firing_rate_matrix_dict, context.template_binned_t_edges,
                                              duration=context.template_duration, sorted_gids=context.sorted_gid_dict)


    context.interface.update_worker_contexts(sorted_gid_dict=context.sorted_gid_dict,
                                             template_firing_rate_matrix_dict=context.template_firing_rate_matrix_dict)

    if context.disp:
        print('decode_simple_network_heldout_run: processing template data for %i trials took %.1f s' %
              (len(context.template_trial_keys), time.time() - start_time))
        sys.stdout.flush()
    current_time = time.time()

    if not context.decode_data_is_processed:
        # return dicts to analyze
        decoded_pos_matrix_dict = \
            decode_data(context.decode_data_file_path, context.decode_trial_keys, context.decode_group_key,
                        context.export_data_key, context.decode_window_dur, context.decode_duration,
                        context.template_window_dur, context.template_duration, export=context.export,
                        disp=context.disp, plot=context.plot, plot_trial_keys=context.plot_decode_trial_keys)
    else:
        decoded_pos_matrix_dict = load_decoded_data(context.decode_data_file_path, context.export_data_key)
        if plot and context.plot_n_trials > 0:
            discard = decode_data(context.decode_data_file_path, context.plot_decode_trial_keys,
                                  context.decode_group_key, context.export_data_key, context.decode_window_dur,
                                  context.decode_duration, context.template_window_dur, context.template_duration,
                                  export=False, temp_export=False, disp=context.disp, plot=True,
                                  plot_trial_keys=context.plot_decode_trial_keys)

    if context.disp:
        print('decode_simple_network_heldout_run: decoding data for %i trials took %.1f s' %
              (len(context.decode_trial_keys), time.time() - current_time))
        sys.stdout.flush()

    if context.plot:
        context.interface.apply(plt.show)

    if not interactive:
        context.interface.stop()
    else:
        context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)
