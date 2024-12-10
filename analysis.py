import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from modules.preprocessing import preprocessing, get_groundtruth_with_label


"""
Option settings for data analysis.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
FOLDER = 'condition2'
number_of_bins = 50
figure_resolution_in_dpi = 128
figure_font_size = 20


"""
preprocessing generates 4 data.
@params: data folder path, pixel microns, frame rate, cutoff
@output: DataFrame, DataFrame, ndarray, networkx grpah

preprocessing includes below steps.
1. exclude the trajectory where length is shorter than CUTOFF
2. conver from pixel unit to micrometer unit with PIXELMICRONS and FRAMERATE
3. generate 2 DataFrames, 1 ndarray representation of markovchain, 1 graph respresentation of markovchain
"""
#analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, states = preprocessing(folder=FOLDER, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)
analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, states = get_groundtruth_with_label(folder=FOLDER, label_folder='dummy', pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)


"""
From here, we treat data to make plots or print results.
Data is stored as
1. analysis_data1(DataFrame: contains data of mean_jump_distance, K, alpha, state, length, traj_id)
2. analysis_data2(DataFrame: contains data of displacments, state)
3. state_markov(matrix: contains transition probability)
4. state_graph(network: built from transitions between states(weight: nb of occurence of transitions))
5. msd(DataFrame: contains msd for each state.) 
6. tamsd(DataFrame: contains ensemble-averaged tamsd for each state.) 
-> ref: https://www.researchgate.net/publication/352833354_Characterising_stochastic_motion_in_heterogeneous_media_driven_by_coloured_non-Gaussian_noise
-> ref: https://arxiv.org/pdf/1205.2100
7. classified states beforehand with BI-ADD or other tools.

Units: 
K: generalized diffusion coefficient, um^2/s^alpha
alpha: anomalous diffusion exponent, real number between 0 and 2
mean_jump_disatnce: average of jump distances of single trajectory
state: states defined in BI-ADD
length: length of trajectory, second
displacements: displacements of all trajectories, um
"""
print(f'\nanalysis_data1:\n', analysis_data1)
print(f'\nanalysis_data2:\n', analysis_data2)
print(f'\nMSD:\n', msd)
print(f'\nEnsemble-averaged TAMSD:\n', tamsd)


#p1: kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1', dpi=figure_resolution_in_dpi)
p1 = sns.histplot(analysis_data1, x=f'mean_jump_d', stat='percent', hue='state', bins=number_of_bins, kde=True)
plt.xlabel(f'mean_jump_distance')
p1.set_title(f'mean_jump_distance for each state')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p2: joint distribution plot(kde) of alpha(x-axis) and K(y-axis) for each state
p2 = sns.jointplot(data=analysis_data1, x=f"alpha", y=f"K", kind='kde', hue='state')
plt.xlabel(f'alpha')
plt.ylabel(f'K')
p2.fig.suptitle(f'alpha, K distribution for each state')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p3: histogram of states
plt.figure(f'p3', dpi=figure_resolution_in_dpi)
p3 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='state')
p3.set_title(f'population of states')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p4: state transition probability
plt.figure(f'p4', dpi=figure_resolution_in_dpi)
p4 = sns.heatmap(state_markov, annot=True)
p4.set_title(f'state transition probability')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p5: displacement histogram
plt.figure(f'p5', dpi=figure_resolution_in_dpi)
p5 = sns.histplot(data=analysis_data2, x='displacements', stat='percent', hue='state', bins=number_of_bins, kde=True)
p5.set_title(f'displacement histogram')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p6: trajectory length(frame) histogram
plt.figure(f'p6', dpi=figure_resolution_in_dpi)
p6 = sns.histplot(data=analysis_data1, x='length', stat='percent', hue='state', bins=number_of_bins, kde=True)
p6.set_title(f'trajectory length histogram')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)


#p7: MSD
plt.figure(f'p7', dpi=figure_resolution_in_dpi)
p7 = sns.lineplot(data=msd, x=np.log10(msd['time']), y=np.log10(msd['mean']), hue='state')
p7.set_title(f'log-log MSD')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
for state_idx, state in enumerate(states):
    # lower, upper bound related to the number of data (TODO: testing now)
    msd_per_state = msd[msd['state'] == state].sort_values('time')
    lower_bound = [mu - sigma for mu, sigma in zip(msd_per_state['mean'], msd_per_state['mean'] / 4 * (1 / msd_per_state['nb_data'] / (1 / msd_per_state['nb_data']).max()))]
    upper_bound = [mu + sigma for mu, sigma in zip(msd_per_state['mean'], msd_per_state['mean'] / 4 * (1 / msd_per_state['nb_data'] / (1 / msd_per_state['nb_data']).max()))]
    plt.fill_between(msd_per_state['time'], lower_bound, upper_bound, alpha=.3, color=f'C{state_idx}')
plt.xticks(rotation=90)


#p8: Ensemble-averaged TAMSD
plt.figure(f'p8', dpi=figure_resolution_in_dpi)
p8 = sns.lineplot(data=tamsd, x='time', y='mean', hue='state')
p8.set_title(f'log-log Ensemble-averaged TAMSD')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
for state_idx, state in enumerate(states):
    # lower, upper bound related to the number of data (TODO: testing now)
    tamsd_per_state = tamsd[tamsd['state'] == state].sort_values('time')
    lower_bound = [mu - sigma for mu, sigma in zip(tamsd_per_state['mean'], tamsd_per_state['mean'] / 4 * (1 / tamsd_per_state['nb_data'] / (1 / tamsd_per_state['nb_data']).max()))]
    upper_bound = [mu + sigma for mu, sigma in zip(tamsd_per_state['mean'], tamsd_per_state['mean'] / 4 * (1 / tamsd_per_state['nb_data'] / (1 / tamsd_per_state['nb_data']).max()))]
    plt.fill_between(tamsd_per_state['time'], lower_bound, upper_bound, alpha=.3, color=f'C{state_idx}')
plt.xticks(rotation=90)


plt.show()
