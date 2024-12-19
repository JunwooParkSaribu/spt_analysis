import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from module.preprocessing import preprocessing
from module.fileIO.DataLoad import read_multiple_csv, read_multiple_h5s


"""
Option settings for data analysis.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
CONDITIONS = ['condition1', 'condition2']
number_of_bins = 50
figure_resolution_in_dpi = 200
figure_font_size = 20


"""
We concatenate different conditions of Dataframe into one for further analysis.
"""
analysis_data1 = pd.DataFrame({})
analysis_data2 = pd.DataFrame({})
state_markovs = []
state_graphs = []
for condition in CONDITIONS:
    data = read_multiple_h5s(path=condition)
    data1, data2, state_markov, state_graph, msd, tamsd, states = preprocessing(data=data, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF, tamsd_calcul=False)
    data1['condition'] = [condition] * len(data1)
    data2['condition'] = [condition] * len(data2)
    analysis_data1 = pd.concat([analysis_data1, data1], ignore_index=True)
    analysis_data2 = pd.concat([analysis_data2, data2], ignore_index=True)
    state_markovs.append(state_markov)
    state_graphs.append(state_graph)
    


"""
From here, we treat data to make plots or print results.
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


#p2: histogram of states
plt.figure(f'p2', dpi=figure_resolution_in_dpi)
p2 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='condition', common_norm=False)
p2.set_title(f'population of states')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
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


plt.show()
