import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from module.preprocessing import preprocessing
from module.fileIO.DataLoad import read_multiple_csv, read_multiple_h5s
from scipy.stats import bootstrap
import numpy as np


"""
Option settings for data analysis. Same as analysis.py script but for the comparison of multiple conditions.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
CONDITIONS = ['condition1', 'condition2']
number_of_bins = 50
bootstrap_bins = 300
figure_resolution_in_dpi = 200
figure_font_size = 20
y_lim_for_percent = [0, 20]
x_lim_for_mean_jump_distances = [0, 0.5]


"""
We concatenate different conditions of Dataframe into one for further analysis.
"""
analysis_data1 = pd.DataFrame({})
analysis_data2 = pd.DataFrame({})
state_markovs = []
state_graphs = []
for condition in CONDITIONS:
    data = read_multiple_h5s(path=condition)
    data1, data2, state_markov, state_graph, msd, tamsd, states, state_changing_duration = preprocessing(data=data, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF, tamsd_calcul=False)
    data1['condition'] = [condition] * len(data1)
    data2['condition'] = [condition] * len(data2)
    analysis_data1 = pd.concat([analysis_data1, data1], ignore_index=True)
    analysis_data2 = pd.concat([analysis_data2, data2], ignore_index=True)
    state_markovs.append(state_markov)
    state_graphs.append(state_graph)
    

"""
From here, we treat the data to make plots or print results.
Data is stored as
1.analysis_data1: (DataFrame: contains data of mean_jump_distance, log10_K, alpha, state, duration, traj_id, condition)
2.analysis_data2: (DataFrame: contains data of displacments, state, condition)
3.state_markovs: (matrix: contains transition probability of conditions)
4.state_graphs: (network: built from transitions between states(weight: nb of occurence of transitions) of conditions)
"""
print(f'\nanalysis_data1:\n', analysis_data1)
print(f'\nanalysis_data2:\n', analysis_data2)


#p1: histogram with kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1', dpi=figure_resolution_in_dpi)
p1 = sns.histplot(data=analysis_data1, x=f'mean_jump_d', stat='percent', hue='condition', common_norm=False, bins=number_of_bins, kde=True)
p1.set_xlabel(r'mean jump-distance($\mu m$)')
p1.set_title(f'mean jump-distance for each condition')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.tight_layout()


#p2: population of each state per condition
state_population = []
state_per_condition = {}
for cd in analysis_data1['condition'].unique():
    state_per_condition[cd] = {}
    cond_data = analysis_data1[analysis_data1['condition']==cd]
    for st in cond_data['state'].unique():
        state_per_condition[cd][st] = len(cond_data[cond_data['state'] == st]) / len(cond_data)
for st, cd in zip(np.array(analysis_data1['state']), np.array(analysis_data1['condition'])):
    state_population.append(state_per_condition[cd][st])
analysis_data1['state_population'] = state_population
p2 = sns.catplot(data=analysis_data1, x="condition", y="state_population", hue='state', kind="bar", height=12)
p2.set_axis_labels(fontsize=figure_font_size)
p2.figure.suptitle(r'Population of each state per condition')
plt.tight_layout()


#p3: displacement histogram
plt.figure(f'p3', dpi=figure_resolution_in_dpi)
p3 = sns.histplot(data=analysis_data2, x='displacements', stat='percent', hue='condition', common_norm=False, bins=number_of_bins, kde=True)
p3.set_title(f'displacement histogram')
p3.set_xlabel(r'displacment($\mu m$)')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.tight_layout()


#p4: bootstrapped distribution with kde(kernel density estimation) plot for averaged mean jump-distances grouped by state.
plt.figure(f'p4', dpi=figure_resolution_in_dpi)
bootstrapped_data = {'averaged_mean_jump_distances':[], 'state':[], 'condition':[]}
bootstrapped_results = []
for cd in analysis_data1['condition'].unique():
    cond_data = analysis_data1[analysis_data1['condition']==cd]
    for st in cond_data['state'].unique():
        bts = bootstrap([np.array(cond_data[cond_data['state'] == st]['mean_jump_d'])], np.mean, n_resamples=1000, confidence_level=0.95)
        bootstrapped_data['averaged_mean_jump_distances'].extend(bts.bootstrap_distribution)
        bootstrapped_data['state'].extend([st] * len(bts.bootstrap_distribution))
        bootstrapped_data['condition'].extend([cd] * len(bts.bootstrap_distribution))
        bootstrapped_results.append(bts)
p4 = sns.histplot(bootstrapped_data, x=f'averaged_mean_jump_distances', stat='percent', hue='condition', bins=bootstrap_bins, kde=False)
p4.set_xlabel(r'bootstrapped mean jump-distances($\mu m$)')
p4.set_title(f'bootstrapped mean jump-distances for each state')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()


plt.show()
