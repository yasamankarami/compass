# Created by sneha bheemireddy at 12/8/24
import configparser
import os
from argparse import Namespace

allowed_params = {

    'generals': {
        'topology',
        'trajectory',
        'output_dir',
        'n_cores',
        'job_name'},

    'non_bond': {
        'non_bond_cut'},

    'salt_bridges': {
        'NO_cut'},

    'hbonds': {
        'DA_cut',
        'HA_cut',
        'DHA_cut',
        'heavy'},

    'networks': {
        'pdb_file_path',
        'min_dist_matrix_file',
        'adjacency_file',
        'results_dir'}

}

allowed_heavies = {'S', 'N', 'O'}


def read_config_file(config_path):
    """
    Read configuration file using argparse

    Args:
        config_path: path to the config file

    Returns:
        config_obj: configparser read object
    """
    config_obj = configparser.ConfigParser(allow_no_value=True,
                                           inline_comment_prefixes='#')
    config_obj.optionxform = str
    config_obj.read(config_path)
    return config_obj


def check_config(config_obj):
    """
    Check sections and values of a given configuration file

    Args:
        config_obj: configparser read object

    Returns:
        config_dict: a dict of raw parameters as specified in the config file
    """
    read_sections = set(config_obj.sections())
    allowed_sections = set(allowed_params.keys())
    sections_equals = read_sections == allowed_sections

    # Check sections
    if not sections_equals:
        raise ValueError(f'\nIncongruence in the number or naming of declared'
                         f' sections. Only the following are supported: {allowed_sections}')

    # Check keys
    config_dict = {}
    for section in allowed_params:
        allowed_keys = allowed_params[section]
        read_keys = config_obj[section]
        same_keys = set(read_keys) == allowed_keys
        if not same_keys:
            raise ValueError(f'\nIncongruence in the number or naming of'
                             f' declared keys. Only the following are '
                             f'supported for section [{section}]: {allowed_keys}')

        # Update config
        section_items = dict(config_obj[section].items())
        config_dict.update({section: section_items})

    return config_dict


def parse_params(config_path):
    """
    Parse parameters contained in a configuration file.

    Args:
        config_path: Path to the config file.

    Returns:
        param_space: argparse.Namespace containing all parsed parameters.
    """
    config_obj = read_config_file(
        config_path)  # Assumes this function reads and parses the config file
    param_dict = check_config(
        config_obj)  # Assumes this function checks and validates the config
    param_space = Namespace()

    # General params
    param_space.topo = param_dict['generals']['topology']
    param_space.traj = param_dict['generals']['trajectory']
    param_space.out_dir = param_dict['generals']['output_dir']
    param_space.n_cores = int(param_dict['generals']['n_cores'])
    param_space.title = param_dict['generals']['job_name']

    # Descriptor params
    param_space.nb_cut = float(param_dict['non_bond']['non_bond_cut'])
    param_space.sb_cut = float(param_dict['salt_bridges']['NO_cut'])
    param_space.da_cut = float(param_dict['hbonds']['DA_cut'])
    param_space.ha_cut = float(param_dict['hbonds']['HA_cut'])
    param_space.dha_cut = float(param_dict['hbonds']['DHA_cut'])
    param_space.heavies = set(param_dict['hbonds']['heavy'].split())

    allowed_heavies = {'N', 'O',
                       'S'}  # Assuming these are the allowed heavy atoms for H-bonds
    if not allowed_heavies.issuperset(param_space.heavies):
        raise ValueError(f'The computing of hbonds considers only '
                         f'{allowed_heavies} as heavy atoms (donor or acceptor).')

    # Network params
    param_space.pdb_file_path = param_dict['networks']['pdb_file_path']
    param_space.min_dist_matrix_file = param_dict['networks'][
        'min_dist_matrix_file']
    param_space.adjacency_file = param_dict['networks']['adjacency_file']
    param_space.results_dir = param_dict['networks']['results_dir']

    os.makedirs(param_space.out_dir, exist_ok=True)
    return param_space, param_dict
