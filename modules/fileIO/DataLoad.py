import os
import csv
import numpy as np
from modules.histone.H2B import H2B
from itertools import islice
from modules.physics import TrajectoryPhy


def read_file(file: str, cutoff: int, filetype='trxyt') -> dict:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    histones = {}
    trajectory = {}
    time = {}

    # Check filetype.
    assert file.strip().split('.')[-1] == filetype

    # Read file and store the trajectory and time information in H2B object
    try:
        with open(file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')
        file_name = file.strip().split('\\')[-1].split('/')[-1].strip()
        for line in lines:
            temp = line.split('\t')
            if 'merged_cells' in file_name:
                key = temp[4].strip() + '@' + temp[0].strip()  # filename + h2b_id
            else:
                key = file_name + '@' + temp[0].strip()  # filename + h2b_id
            x_pos = float(temp[1].strip())
            y_pos = float(temp[2].strip())
            time_step = float(temp[3].strip())

            if key in trajectory:
                trajectory[key].append([x_pos, y_pos])
                time[key].append(time_step)
            else:
                trajectory[key] = [[x_pos, y_pos]]
                time[key] = [time_step]

        for histone in trajectory:
            if len(trajectory[histone]) >= cutoff:
                histones[histone] = H2B()
                histones[histone].set_trajectory(np.array(trajectory[histone]))
                histones[histone].set_time(np.array(time[histone]))
                info = histone.strip().split('@')
                histones[histone].set_id(info[-1])
                histones[histone].set_file_name('@'.join(info[:-1]))

        del trajectory
        del time

        TrajectoryPhy.calcul_max_radius(histones)
        TrajectoryPhy.diff_coef(histones)
        return histones
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def read_files(paths: list, cutoff=8, group_size=160, chunk=True) -> list:
    """
    @params : paths(list, can be directory or multiple trxyt files), cutoff value(Integer),
              group_size(Integer), chunk(boolean)
    @return : list of splitted dictionary containing H2B objects
    Read all trajectory files under the given path (can be multiple trajectory files or a path)
    If chunk is true, split the data(dictionary) into a list for a group size to control the memory usage.
    Else, return a list containing a single dictionary.
    group_size decide the memory size when the trajectory files convert into images.
    (large group size increase the speed but takes large amount of memory)
    """
    histones = {}
    split_histones = []
    # If the given path is directory
    if os.path.isdir(paths[0]):
        files = os.listdir(paths[0])
        if len(files) > 0:
            for file in files:
                if 'trxyt' in file:
                    h = read_file(paths[0] + '/' + file, cutoff=cutoff, filetype='trxyt')
                    histones |= h
        if not chunk:
            return [histones]
        for item in chunks(histones, group_size):
            split_histones.append(item)
        return split_histones
    # If the given path contains multiple files
    else:
        for file in paths:
            h = read_file(file, cutoff=cutoff)
            histones |= h
        if not chunk:
            return [histones]
        for item in chunks(histones, group_size):
            split_histones.append(item)
        return split_histones


def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def read_report(file: str):
    """
    @params : filename(String)
    @return : header(dict), lines(dict)
    Read a report file(csv), return the header and lines
    """
    lines = []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        for row in reader:
            lines.append(row)
    return header, lines


def read_params(path: str) -> dict:
    """
    @params : path(String)
    @return : parameters(dict)
    Read a configuration file in a given path and return the parameters.
    """
    file = f'{path}/config.txt'
    params = {}

    with open(file, 'r') as f:
        input = f.readlines()
        for line in input:
            if '=' not in line:
                continue
            line = line.strip().split('=')
            param_name = line[0].strip()
            param_val = line[1].split('#')[0].strip()
            if param_name in ['amp', 'nChannel', 'batch_size', 'group_size', 'cut_off']:
                params[param_name] = int(param_val)
            elif param_name in ['data']:
                if param_name in params:
                    params[param_name].append(param_val)
                else:
                    params[param_name] = [param_val]
            elif param_name in ['all'] or param_name in ['makeImage'] or param_name in ['postProcessing']:
                if param_val.lower() == 'true' or param_val.lower() == 'yes' or param_val == '1':
                    params[param_name] = True
                else:
                    params[param_name] = False
            else:
                params[param_name] = param_val

    # if given data is single trxyt file, set parameter "all" to False to avoid unnecessary conflict
    if len(params['data']) == 1 and params['all'] and '.trxyt' in params['data'][0]:
        params['all'] = False
    return params
