import click
from nested.parallel import get_parallel_interface
from nested.utils import get_unknown_click_arg_dict
from nested.optimize_utils import Context
from simple_network_analysis_utils import *

context = Context()

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              default='data')
@click.option("--export-data-key", type=str, default='0')
@click.option("--template-group-key", type=str, default='simple_network_exported_run_data')
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--verbose", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config_file_path, data_dir, export_data_key, template_group_key, decode_window_dur, verbose, debug):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param template_group_key: str
    :param decode_window_dur: float (ms)
    :param verbose: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())

    if config_file_path is None or not os.path.isfile(config_file_path):
        raise RuntimeError('analyze_simple_network_decoded_replay_instances: invalid config_file_path: %s' %
                           config_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(config_dict)

    if 'replay_data_file_name_list' not in context():
        raise RuntimeError('analyze_simple_network_decoded_replay_instances: config_file_path: %s does not'
                           'contain required replay_data_file_name_list' % config_file_path)
    if 'template_data_file_name_list' not in context():
        raise RuntimeError('analyze_simple_network_decoded_replay_instances: config_file_path: %s does not'
                           'contain required template_data_file_name_list' % config_file_path)
    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None

    template_data_file_path = data_dir + '/' + context.template_data_file_name_list[0]
    if not os.path.isfile(template_data_file_path):
        raise IOError('analyze_simple_network_decoded_replay_instances: invalid template_data_file_path: %s' %
                      template_data_file_path)

    shared_context_key = 'shared_context'
    with h5py.File(template_data_file_path, 'r') as f:
        group = get_h5py_group(f, [export_data_key, template_group_key])
        subgroup = get_h5py_group(group, [shared_context_key])
        template_duration = get_h5py_attr(subgroup.attrs, 'duration')

    decoded_pos_matrix_dict_list = []
    for replay_data_file_name in context.replay_data_file_name_list:
        replay_data_file_path = data_dir + '/' + replay_data_file_name
        if not os.path.isfile(replay_data_file_path):
            raise IOError('analyze_simple_network_decoded_replay_instances: invalid replay_data_file_path: %s' %
                          replay_data_file_path)
        decoded_pos_matrix_dict = load_decoded_data(replay_data_file_path, export_data_key)
        decoded_pos_matrix_dict_list.append(decoded_pos_matrix_dict)
    plot_decoded_trajectory_replay_data(decoded_pos_matrix_dict_list, bin_dur=decode_window_dur,
                                        template_duration=template_duration, pop_order=context.pop_order,
                                        label_dict=context.label_dict, color_dict=context.color_dict)
    plt.show()

    context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
