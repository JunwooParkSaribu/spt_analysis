import pandas as pd
import numpy as np
import glob
from module.TrajectoryObject import TrajectoryObj


def read_h5(file):
    with pd.HDFStore(file) as hdf_store:
        metadata = hdf_store.get_storer('data').attrs.metadata
        df_read = hdf_store.get('data')
    df_read = df_read.dropna()
    convert_dict = {'state': int, 'frame': int, 'traj_idx': int}
    df_read = df_read.astype(convert_dict)
    return df_read, metadata


def read_csv(file):
    csv_data = pd.read_csv(file)
    col_names = ['traj_idx', 'frame', 'x', 'y', 'z', 'state', 'K', 'alpha']
    z = np.zeros(len(csv_data.iloc[:, 1]))
    state = np.zeros(len(csv_data.iloc[:, 1]), dtype=np.int32)
    K = np.zeros(len(csv_data.iloc[:, 1]))
    alpha = np.zeros(len(csv_data.iloc[:, 1]))
    csv_data = csv_data.assign(z = z)
    csv_data = csv_data.assign(state = state)
    csv_data = csv_data.assign(K = K)
    csv_data = csv_data.assign(alpha = alpha)
    return csv_data


def read_multiple_h5s(path:str):
    dfs = []
    meta_info = []
    files_not_same_conditions = []
    prefix = f'_biadd'

    if path.split('.')[-1] == 'h5':
        f_list = [path]
    else:
        f_list = glob.glob(f'{path}/*{prefix}.h5')
    for f_idx, file in enumerate(f_list):
        try:
            df, meta = read_h5(file)
            if f_idx == 0:
                meta_info.append(meta['sample_id'])

            if meta['sample_id'] not in meta_info:
                files_not_same_conditions.append(file)

            pure_f_name = file.split('/')[-1].split(f'{prefix}.h5')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
            dfs.append(df)
        except:
            print(f'*** No trajectory data is found in the file: {file}, skipping this file. ***')

    grouped_df = pd.concat(dfs)

    if len(files_not_same_conditions) > 1:
        print('********************************************************************************************************************************************************')
        print("Below files were processed in different folders or seperately while the prediction with BI-ADD even though they are grouped for a same condition in the analysis.")
        print("Next time, it would be recommended to run BI-ADD with placing these files under a single folder to avoid unexpcted errors/bias if they have same condition.")
        for ff in files_not_same_conditions:
            print(ff)
        print('********************************************************************************************************************************************************')
    return grouped_df


def read_multiple_csv(path):
    dfs = []
    f_list = glob.glob(f'{path}/*.csv')
    for f_idx, file in enumerate(f_list):
        df = read_csv(file)
        if f_idx != 0:
            pure_f_name = file.split('/')[-1].split(f'.csv')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        else:
            pure_f_name = file.split('/')[-1].split(f'.csv')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        dfs.append(df)
    grouped_df = pd.concat(dfs) 
    return grouped_df


def andi2_label_parser(path):
    andi_dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\n')[0].split(',')
            traj_idx = line[0]
            Ks = []
            alphas = []
            states = []
            cps = []
            for K, alpha, state, cp in np.array(line[1:], dtype=object).reshape(-1, 4):
                Ks.append(float(K))
                alphas.append(float(alpha))
                states.append(int(eval(state)))
                cps.append(int(eval(cp)))
            andi_dict[f'{path.split(".txt")[0].split("/")[-1]}@{traj_idx}'] = np.array([Ks, alphas, states, cps]).T
    return andi_dict


def read_mulitple_andi_labels(path):
    andi_dicts = {}
    prefix = 'fov_*'
    f_list = glob.glob(f'{path}/*{prefix}.txt')
    for f_idx, file in enumerate(f_list):
        andi_dict = andi2_label_parser(file)
        andi_dicts |= andi_dict
    return andi_dicts


def read_trajectory(file: str, andi_gt=False, pixel_microns=1.0, frame_rate=1.0) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx', 'csv']
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        localizations = {}
        tmp = {}
        try:
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            for line in lines:
                temp = line.split('\t')
                x_pos = float(temp[1].strip()) * pixel_microns
                y_pos = float(temp[2].strip()) * pixel_microns
                z_pos = 0. * pixel_microns
                time_step = float(temp[3].strip()) * frame_rate
                if time_step in tmp:
                    tmp[time_step].append([x_pos, y_pos, z_pos])
                else:
                    tmp[time_step] = [[x_pos, y_pos, z_pos]]

            time_steps = np.sort(np.array(list(tmp.keys())))
            first_frame, last_frame = time_steps[0], time_steps[-1]
            steps = np.arange(int(np.round(first_frame * 100)), int(np.round(last_frame * 100)) + 1)
            for step in steps:
                if step/100 in tmp:
                    localizations[step] = tmp[step/100]
                else:
                    localizations[step] = []
            return localizations
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
    else:
        try:
            trajectory_list = []
            with open(file, 'r', encoding="utf-8") as f:
                input = f.read()
            lines = input.strip().split('\n')
            nb_traj = 0
            old_index = -999
            for line in lines[1:]:
                temp = line.split(',')
                index = int(float(temp[0].strip()))
                frame = int(float(temp[1].strip()))
                x_pos = float(temp[2].strip())
                y_pos = float(temp[3].strip())
                if andi_gt:
                    x_pos = float(temp[3].strip())
                    y_pos = float(temp[2].strip())
                if len(temp) > 4:
                    z_pos = float(temp[4].strip())
                else:
                    z_pos = 0.0

                if index != old_index:
                    nb_traj += 1
                    trajectory_list.append(TrajectoryObj(index=index, max_pause=5))
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                else:
                    trajectory_list[nb_traj - 1].add_trajectory_position(frame * frame_rate, x_pos * pixel_microns, y_pos * pixel_microns, z_pos * pixel_microns)
                old_index = index
            return trajectory_list
        except Exception as e:
            print(f"Unexpected error, check the file: {file}")
            print(e)
