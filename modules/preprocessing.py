import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from modules.fileIO import DataLoad


def preprocessing(folder, pixelmicrons, framerate, cutoff):
    # load FreeTrace+Bi-ADD data without NaN (NaN where trajectory length is shorter than 5, default in BI-ADD)
    data = DataLoad.read_multiple_h5s(folder).dropna()
    # using dictionary to convert specific columns
    convert_dict = {'state': int}
    data = data.astype(convert_dict)
    traj_indices = pd.unique(data['traj_idx'])


    # initializations
    total_states = sorted(data['state'].unique())
    product_states = list(product(total_states, repeat=2))
    state_graph = nx.DiGraph()
    state_graph.add_nodes_from(total_states)
    state_graph.add_edges_from(product_states, weight=0)
    state_markov = [[0 for _ in range(len(total_states))] for _ in range(len(total_states))]
    analysis_data1 = {}
    analysis_data1[f'mean_jump_d'] = []
    analysis_data1[f'K'] = []
    analysis_data1[f'alpha'] = []
    analysis_data1[f'state'] = []
    analysis_data1[f'length'] = []
    analysis_data1[f'traj_id'] = []
    analysis_data2 = {}
    analysis_data2[f'displacements'] = []
    analysis_data2[f'state'] = []


    # get data from trajectories
    for traj_idx in traj_indices:
        single_traj = data.loc[data['traj_idx'] == traj_idx].copy()
        
        # calculate state changes inside single trajectory
        before_st = single_traj.state.iloc[0]
        for st in single_traj.state:
            state_graph[before_st][st]['weight'] += 1
            before_st = st

        # chucnk into sub-trajectories
        before_st = single_traj.state.iloc[0]
        chunk_idx = [0, len(single_traj)]
        for st_idx, st in enumerate(single_traj.state):
            if st != before_st:
                chunk_idx.append(st_idx)
            before_st = st
        chunk_idx = sorted(chunk_idx)

        for i in range(len(chunk_idx) - 1):
            sub_trajectory = single_traj.iloc[chunk_idx[i]:chunk_idx[i+1]].copy()

            # trajectory length filter condition
            if len(sub_trajectory) >= cutoff:
                # convert from pixel-coordinate to micron.
                sub_trajectory.x *= pixelmicrons
                sub_trajectory.y *= pixelmicrons
                sub_trajectory.z *= pixelmicrons ## need to check
                sub_trajectory.K *= (pixelmicrons**2/framerate)

                # coordinate normalize
                sub_trajectory.x -= sub_trajectory.x.iloc[0]
                sub_trajectory.y -= sub_trajectory.y.iloc[0]

                # calcultae jump distances
                jump_distances = np.sqrt((sub_trajectory.x.iloc[1:].to_numpy() - sub_trajectory.x.iloc[:-1].to_numpy()) ** 2 + (sub_trajectory.y.iloc[1:].to_numpy() - sub_trajectory.y.iloc[:-1].to_numpy()) ** 2)

                # add data for the visualization
                analysis_data1[f'mean_jump_d'].append(jump_distances.mean())
                analysis_data1[f'K'].append(sub_trajectory.K.iloc[0])
                analysis_data1[f'alpha'].append(sub_trajectory.alpha.iloc[0])
                analysis_data1[f'state'].append(sub_trajectory.state.iloc[0])
                analysis_data1[f'length'].append((sub_trajectory.frame.iloc[-1] - sub_trajectory.frame.iloc[0] + 1) * framerate)
                analysis_data1[f'traj_id'].append(sub_trajectory.traj_idx.iloc[0])

                analysis_data2[f'displacements'].extend(list(jump_distances))
                analysis_data2[f'state'].extend([sub_trajectory.state.iloc[0]] * len(list(jump_distances)))

    # normalize markov chain
    for edge in state_graph.edges:
        src, dest = edge
        weight = state_graph[src][dest]["weight"]
        state_markov[src][dest] = weight
    state_markov = np.array(state_markov, dtype=np.float64)
    for idx in range(len(total_states)):
        state_markov[idx] /= np.sum(state_markov[idx])


    analysis_data1 = pd.DataFrame(analysis_data1).astype({'state': int, 'length': float, 'traj_id':str})
    analysis_data2 = pd.DataFrame(analysis_data2)

    return analysis_data1, analysis_data2, state_markov, state_graph
