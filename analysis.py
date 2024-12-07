import seaborn as sns
import matplotlib.pyplot as plt
from modules.preprocessing import preprocessing, get_groundtruth_with_label


"""
Option settings for data analysis.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
FOLDER = 'condition1'
number_of_bins = 50
figure_resolution_in_dpi = 128


"""
preprocessing generates 4 data.
@params: data folder path, pixel microns, frame rate, cutoff
@output: DataFrame, DataFrame, ndarray, networkx grpah

preprocessing includes below steps.
1. exclude the trajectory where length is shorter than CUTOFF
2. conver from pixel unit to micrometer unit with PIXELMICRONS and FRAMERATE
3. generate 2 DataFrames, 1 ndarray representation of markovchain, 1 graph respresentation of markovchain
"""
analysis_data1, analysis_data2, state_markov, state_graph = preprocessing(folder=FOLDER, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)
#analysis_data1, analysis_data2, state_markov, state_graph = get_groundtruth_with_label(folder=FOLDER, label_folder='dummy', pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)


"""
From here, we treat data to make plots or print results.
Data is stored as
1.analysis_data1(DataFrame: contains data of mean_jump_distance, K, alpha, state, length, traj_id)
2.analysis_data2(DataFrame: contains data of displacments, state)
3.state_markov(matrix: contains transition probability)
4.state_graph(network: built from transitions between states(weight: nb of occurence of transitions))

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


#p1: kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1', dpi=figure_resolution_in_dpi)
p1 = sns.histplot(analysis_data1, x=f'mean_jump_d', stat='percent', hue='state', bins=number_of_bins, kde=True)
plt.xlabel(f'mean_jump_distance')
p1.set_title(f'mean_jump_distance for each state')


#p2: joint distribution plot(kde) of alpha(x-axis) and K(y-axis) for each state
p2 = sns.jointplot(data=analysis_data1, x=f"alpha", y=f"K", kind='kde', hue='state')
plt.xlabel(f'alpha')
plt.ylabel(f'K')
p2.fig.suptitle(f'alpha, K distribution for each state')


#p3: histogram of states
plt.figure(f'p3', dpi=figure_resolution_in_dpi)
p3 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='state')
p3.set_title(f'population of states')


#p4: state transition probability
plt.figure(f'p4', dpi=figure_resolution_in_dpi)
p4 = sns.heatmap(state_markov, annot=True)
p4.set_title(f'state transition probability')


#p5: displacement histogram
plt.figure(f'p5', dpi=figure_resolution_in_dpi)
p5 = sns.histplot(data=analysis_data2, x='displacements', stat='percent', hue='state', bins=number_of_bins, kde=True)
p5.set_title(f'displacement histogram')


#p6: trajectory length(frame) histogram
plt.figure(f'p6', dpi=figure_resolution_in_dpi)
p6 = sns.histplot(data=analysis_data1, x='length', stat='percent', hue='state', bins=number_of_bins, kde=True)
p6.set_title(f'trajectory length histogram')


plt.show()
