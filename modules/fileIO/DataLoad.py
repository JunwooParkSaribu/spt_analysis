import os
import csv
import pandas as pd
import numpy as np
from itertools import islice
import glob


def read_h5(file):
    with pd.HDFStore(file) as hdf_store:
        metadata = hdf_store.get_storer('data').attrs.metadata
        df_read = hdf_store.get('data')
    return df_read, metadata


def read_multiple_h5s(path):
    dfs = []
    meta_info = []
    files_not_same_conditions = []

    f_list = glob.glob(f'{path}/*_traces_biadd.h5')
    for f_idx, file in enumerate(f_list):
        df, meta = read_h5(file)
        if f_idx != 0:
            if meta['sample_id'] not in meta_info:
                files_not_same_conditions.append(file)
                continue
            else:
                pure_f_name = file.split('/')[-1].split('_traces_biadd.h5')[0]
                df['filename'] = [pure_f_name] * len(df['traj_idx'])
                traj_indices = df['traj_idx']
                traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
                df['traj_idx'] = traj_indices
        else:
            meta_info.append(meta['sample_id'])
            pure_f_name = file.split('/')[-1].split('_traces_biadd.h5')[0]
            df['filename'] = [pure_f_name] * len(df['traj_idx'])
            traj_indices = df['traj_idx']
            traj_indices = [f'{pure_f_name}_{idx}' for idx in traj_indices]
            df['traj_idx'] = traj_indices
        dfs.append(df)
    grouped_df = pd.concat(dfs)

    if len(files_not_same_conditions) > 1:
        print('*****************************************************************************************')
        print("Below files are skipped due to their conditions are not same, check metadata of h5 file")
        for ff in files_not_same_conditions:
            print(ff)
        print('*****************************************************************************************')
        
    return grouped_df