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
@click.option("--decode-window-dur", type=float, default=20.)
@click.option("--decoded-pos-error-ymax", type=float, default=None)
@click.option("--sequence-len-ymax", type=float, default=None)
@click.option("--verbose", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config_file_path, data_dir, export_data_key, decode_window_dur, decoded_pos_error_ymax, sequence_len_ymax,
         verbose, debug):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param decode_window_dur: float (ms)
    :param decoded_pos_error_ymax: float
    :param sequence_len_ymax: float
    :param verbose: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())

    if config_file_path is None or not os.path.isfile(config_file_path):
        raise RuntimeError('analyze_simple_network_decoded_heldout_run_instances: invalid config_file_path: %s' %
                           config_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(config_dict)

    if 'data_file_name_list' not in context():
        raise RuntimeError('analyze_simple_network_decoded_heldout_run_instances: config_file_path: %s does not'
                           'contain required data_file_name_list' % config_file_path)
    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None

    group_key = 'simple_network_exported_run_data'
    shared_context_key = 'shared_context'
    processed_group_key = 'simple_network_processed_data'

    decoded_pos_error_mean_dict_list = []
    sequence_len_mean_dict_list = []

    first = True
    for data_file_name in context.data_file_name_list:
        data_file_path = data_dir + '/' + data_file_name
        if not os.path.isfile(data_file_path):
            raise IOError('analyze_simple_network_decoded_heldout_run_instances: invalid data_file_path: %s' %
                          data_file_path)
        with h5py.File(data_file_path, 'r') as f:
            if first:
                group = get_h5py_group(f, [context.export_data_key, group_key])
                subgroup = get_h5py_group(group, [shared_context_key])
                decode_duration = get_h5py_attr(subgroup.attrs, 'duration')
                actual_position = np.arange(0., decode_duration, context.decode_window_dur) + \
                                  context.decode_window_dur / 2.
                first = False

            group = get_h5py_group(f, [context.export_data_key])
            if processed_group_key not in group or shared_context_key not in group[processed_group_key] or \
                    'decoded_pos_matrix' not in group[processed_group_key][shared_context_key]:
                raise RuntimeError('analyze_simple_network_decoded_heldout_run_instances: data_file_path: %s does not'
                                   'contain required decoded_pos_matrix' % data_file_path)
        decoded_pos_matrix_dict = load_decoded_data(data_file_path, context.export_data_key)
        decoded_pos_error_mean_dict, decoded_pos_error_sem_dict, sequence_len_mean_dict, sequence_len_sem_dict = \
            analyze_decoded_trajectory_run_data(decoded_pos_matrix_dict, actual_position, decode_duration)
        decoded_pos_error_mean_dict_list.append(decoded_pos_error_mean_dict)
        sequence_len_mean_dict_list.append(sequence_len_mean_dict)

    decoded_pos_error_mean_dict, decoded_pos_error_sem_dict, sequence_len_mean_dict, sequence_len_sem_dict = \
        analyze_decoded_trajectory_run_data_across_instances(decoded_pos_error_mean_dict_list,
                                                             sequence_len_mean_dict_list)

    plot_decoded_trajectory_run_data(decoded_pos_error_mean_dict, sequence_len_mean_dict, actual_position,
                                     decode_duration, decoded_pos_error_sem_dict, sequence_len_sem_dict,
                                     decoded_pos_error_ymax=decoded_pos_error_ymax,
                                     sequence_len_ymax=sequence_len_ymax, pop_order=context.pop_order,
                                     label_dict=context.label_dict, color_dict=context.color_dict)

    plt.show()

    context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
