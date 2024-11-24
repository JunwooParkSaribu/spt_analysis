import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from modules.fileIO import DataLoad


PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5  # the lowest value is 5


# load FreeTrace+Bi-ADD data without NaN (NaN where trajectory length is shorter than 5, default in BI-ADD)
data = DataLoad.read_multiple_h5s('condition1').dropna()
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
analysis_data = {}
for st in total_states:
    analysis_data[f'mean_jump_d'] = []
    analysis_data[f'K'] = []
    analysis_data[f'alpha'] = []
    analysis_data[f'state'] = []


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
        if len(sub_trajectory) >= CUTOFF:
            # convert from pixel-coordinate to micron.
            sub_trajectory.x *= PIXELMICRONS
            sub_trajectory.y *= PIXELMICRONS
            sub_trajectory.z *= PIXELMICRONS ## need to check
            sub_trajectory.K *= (PIXELMICRONS**2/FRAMERATE)

            # coordinate normalize
            sub_trajectory.x -= sub_trajectory.x.iloc[0]
            sub_trajectory.y -= sub_trajectory.y.iloc[0]

            # calcultae jump distances
            jump_distances = np.sqrt(sub_trajectory.x ** 2 + sub_trajectory.y ** 2)

            # add data for the visualization
            analysis_data[f'mean_jump_d'].append(jump_distances.mean())
            analysis_data[f'K'].append(sub_trajectory.K.iloc[0])
            analysis_data[f'alpha'].append(sub_trajectory.alpha.iloc[0])
            analysis_data[f'state'].append(sub_trajectory.state.iloc[0])


# normalize markov chain
for edge in state_graph.edges:
    src, dest = edge
    weight = state_graph[src][dest]["weight"]
    state_markov[src][dest] = weight
state_markov = np.array(state_markov, dtype=np.float64)
for idx in range(len(total_states)):
    state_markov[idx] /= np.sum(state_markov[idx])



"""
From here, plot functions.
Data is stored in 
1.analysis_data(DataFrame: contains data)
2.state_markov(matrix: contains transition probability)
3.state_graph(network: built from transitions between states(weight: nb of occurence of transitions))
"""
#p1: kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1')
p1 = sns.kdeplot(analysis_data, x=f'mean_jump_d', hue='state')
plt.xlabel(f'mean_jump_distance for each state')
p1.set_title(f'mean_jump_distance')


#p2: joint distribution plot(kde) of alpha(x-axis) and K(y-axis) for each state
p2 = sns.jointplot(data=analysis_data, x=f"alpha", y=f"K", kind='kde', hue='state')
plt.xlabel(f'alpha')
plt.ylabel(f'K')
p2.fig.suptitle(f'alpha, K distribution for each state')

#p3: histogram of states
plt.figure(f'p3')
p3 = sns.histplot(data=analysis_data, x="state", stat='percent', hue='state')
p3.set_title(f'population of states')


#p4: state transition probability
plt.figure(f'p4')
p4 = sns.heatmap(state_markov, annot=True)
p4.set_title(f'state transition probability')
plt.show()

