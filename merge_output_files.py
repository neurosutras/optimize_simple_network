from nested.optimize_utils import *
import click


@click.command()
@click.option("--merge-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option("--output-dir", type=str, default='data')
@click.option("--verbose", is_flag=True)
@click.option("--remove", type=bool, default=True)
def main(merge_file_path, output_dir, verbose, remove):
    """

    :param merge_file_path: str (path to .yaml file)
    :param output_dir: str (path to dir)
    :param verbose: bool
    :param remove: bool
    """
    data = read_from_yaml(merge_file_path)
    if 'export_file_path' in data:
        export_file_path = data['export_file_path']
    else:
        export_file_path = None
    if 'temp_output_path_list' in data:
        temp_output_path_list = data['temp_output_path_list']
    else:
        temp_output_path_list = None
    if temp_output_path_list is None or len(temp_output_path_list) < 1:
        raise RuntimeError('merge_output_files: missing temp_output_path_list')

    merge_hdf5_temp_output_files(temp_output_path_list, export_file_path, output_dir=output_dir, verbose=verbose)

    if remove:
        for temp_output_path in temp_output_path_list:
            os.remove(temp_output_path)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(os.path.basename(__file__)) != -1, sys.argv) + 1):],
         standalone_mode=False)