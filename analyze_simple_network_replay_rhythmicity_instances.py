import click
from nested.parallel import get_parallel_interface
from nested.utils import get_unknown_click_arg_dict
from nested.optimize_utils import Context
from simple_network_analysis_utils import *
from analyze_simple_network_replay_rhythmicity import load_replay_fft_trial_matrix_from_file

context = Context()

@click.command()
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              default='data')
@click.option("--export-data-key", type=str, default='0')
@click.option("--verbose", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config_file_path, data_dir, export_data_key, verbose, debug):
    """

    :param config_file_path: str (path to .yaml file)
    :param data_dir: str (path to dir containing .hdf5 files)
    :param export_data_key: str; top-level key to access data from .hdf5 files
    :param verbose: bool
    :param debug: bool
    """
    # requires a global variable context: :class:'Context'
    context.update(locals())

    if config_file_path is None or not os.path.isfile(config_file_path):
        raise RuntimeError('analyze_simple_network_replay_rhythmicity_instances: invalid config_file_path: %s' %
                           config_file_path)
    config_dict = read_from_yaml(config_file_path)
    context.update(config_dict)

    if 'replay_data_file_name_list' not in context():
        raise RuntimeError('analyze_simple_network_replay_rhythmicity_instances: config_file_path: %s does not'
                           'contain required replay_data_file_name_list' % config_file_path)
    if 'pop_order' not in context():
        context.pop_order = None
    if 'label_dict' not in context():
        context.label_dict = None
    if 'color_dict' not in context():
        context.color_dict = None

    fft_power_mean_dict = {}
    fft_power_sem_dict = {}
    for replay_data_file_name in context.replay_data_file_name_list:
        replay_data_file_path = data_dir + '/' + replay_data_file_name
        if not os.path.isfile(replay_data_file_path):
            raise IOError('analyze_simple_network_replay_rhythmicity_instances: invalid replay_data_file_path: %s' %
                          replay_data_file_path)
        fft_f, fft_power_matrix_dict = \
            load_replay_fft_trial_matrix_from_file(replay_data_file_path, export_data_key)
        for pop_name in fft_power_matrix_dict:
            if pop_name not in fft_power_mean_dict:
                fft_power_mean_dict[pop_name] = []
            this_fft_power_mean = np.mean(fft_power_matrix_dict[pop_name], axis=0)
            fft_power_mean_dict[pop_name].append(this_fft_power_mean)

    for pop_name in fft_power_mean_dict:
        num_instances = len(fft_power_mean_dict[pop_name])
        fft_power_sem_dict[pop_name] = np.std(fft_power_mean_dict[pop_name], axis=0) / \
                                              np.sqrt(num_instances)
        fft_power_mean_dict[pop_name] = np.mean(fft_power_mean_dict[pop_name], axis=0)

    plot_rhythmicity_psd(fft_f, fft_power_mean_dict, fft_power_sem_dict, pop_order=context.pop_order,
                         label_dict=context.label_dict, color_dict=context.color_dict, compressed_plot_format=True,
                         title='Offline rhythmicity')
    plt.show()

    context.update(locals())


if __name__ == '__main__':
    main(standalone_mode=False)
