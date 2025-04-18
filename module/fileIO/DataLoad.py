import pandas as pd
import numpy as np
import glob


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
    if 'Bound' in file:
        state = np.zeros(len(csv_data.iloc[:, 1]), dtype=np.int32)
    elif 'Hybrid' in file:
        state = np.zeros(len(csv_data.iloc[:, 1]), dtype=np.int32) + 1
    elif 'Mobile' in file:
        state = np.zeros(len(csv_data.iloc[:, 1]), dtype=np.int32) + 2
    else:
        state = np.zeros(len(csv_data.iloc[:, 1]), dtype=np.int32) + 3
    K = np.zeros(len(csv_data.iloc[:, 1]))
    alpha = np.zeros(len(csv_data.iloc[:, 1]))
    csv_data = csv_data.assign(z = z)
    csv_data = csv_data.assign(state = state)
    csv_data = csv_data.assign(K = K)
    csv_data = csv_data.assign(alpha = alpha)
    return csv_data


def read_multiple_h5s(path):
    dfs = []
    meta_info = []
    files_not_same_conditions = []
    prefix = f'_biadd'

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
        print("Below files were processed in different folders in BI-ADD even though they are grouped for a same condition in the analysis.")
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
