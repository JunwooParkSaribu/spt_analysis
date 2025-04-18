import seaborn as sns
import matplotlib.pyplot as plt
from module.visuailzation import trajectory_visualization
from module.preprocessing import preprocessing, get_groundtruth_with_label
from module.fileIO.DataLoad import read_multiple_csv, read_multiple_h5s
from scipy.stats import bootstrap
import numpy as np


"""
Option settings for data analysis.
"""
PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
FOLDER = 'condition2'
number_of_bins = 50
bootstrap_bins = 300
figure_resolution_in_dpi = 200
figure_font_size = 20
y_lim_for_percent = [0, 20]
x_lim_for_mean_jump_distances = [0, 0.5]

"""
preprocessing generates 7 data.
@params: data folder path, pixel microns, frame rate, cutoff
@output: DataFrame, DataFrame, ndarray, networkx grpah, DataFrame, DataFrame, list

preprocessing includes below steps.
1. exclude the trajectory where length is shorter than CUTOFF
2. convert from pixel unit to micrometer unit with PIXELMICRONS and FRAMERATE
3. generate 4 DataFrames, 1 ndarray representation of markovchain, 1 graph respresentation of markovchain, 1 list containing states
"""
original_data = read_multiple_h5s(path=FOLDER)
analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, states = preprocessing(data=original_data, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF, tamsd_calcul=False)
#analysis_data1, analysis_data2, state_markov, state_graph, msd, tamsd, states = get_groundtruth_with_label(data=original_data, label_folder='dummy', pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)


"""
From here, we treat data to make plots or print results.
Data is stored as
1. analysis_data1: (DataFrame: contains data of mean_jump_distance, log10_K, alpha, state, duration, traj_id)
2. analysis_data2: (DataFrame: contains data of displacments, state)
3. state_markov: (matrix: contains transition probability)
4. state_graph: (network: built from transitions between states(weight: nb of occurence of transitions))
5. msd: (DataFrame: contains msd for each state.) 
6. tamsd: (DataFrame: contains ensemble-averaged tamsd for each state.) 
-> ref: https://www.researchgate.net/publication/352833354_Characterising_stochastic_motion_in_heterogeneous_media_driven_by_coloured_non-Gaussian_noise
-> ref: https://arxiv.org/pdf/1205.2100
7. states: classified states beforehand with BI-ADD or other tools.

Units: 
log10_K: generalized diffusion coefficient in log10, um^2/s^alpha.
alpha: anomalous diffusion exponent, real number between 0 and 2.
mean_jump_disatnce: set of averages of jump distances in um.
state: states defined in BI-ADD.
duration: duration of trajectory in seconds.
displacement: displacement(time lag=1) of all trajectories in um.
"""
print(f'\nanalysis_data1:\n', analysis_data1)
print(f'\nanalysis_data2:\n', analysis_data2)
print(f'\nMSD:\n', msd)
print(f'\nEnsemble-averaged TAMSD:\n', tamsd)


#p1: histogram with kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1', dpi=figure_resolution_in_dpi)
p1 = sns.histplot(analysis_data1, x=f'mean_jump_d', stat='percent', hue='state', bins=number_of_bins, kde=True)
p1.set_xlabel(r'mean jump-distance($\mu m$)')
p1.set_title(f'mean jump-distance for each state')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()


#p2: joint distribution plot of alpha(x-axis) and K(y-axis) for each state
p2 = sns.jointplot(data=analysis_data1, x=f'alpha', y=f'log10_K', kind='scatter', hue='state', height=12, xlim=(-0.2, 2.2), 
                   joint_kws={'data':analysis_data1, 'size':'duration', 'sizes':(40, 400), 'alpha':0.5})
p2.set_axis_labels(xlabel=r'$\alpha$', ylabel=r'$log_{10}K(\mu m^2/s^\alpha)$', fontsize=figure_font_size)
p2.figure.suptitle(r'$\alpha$, $K$ distribution for each state')
p2.ax_joint.set_yticklabels(p2.ax_joint.get_yticks(), fontsize = figure_font_size)
p2.ax_joint.set_xticklabels(p2.ax_joint.get_xticks(), fontsize = figure_font_size)
plt.tight_layout()


#p3: histogram of states
plt.figure(f'p3', dpi=figure_resolution_in_dpi)
p3 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='state')
p3.set_title(f'population of states')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.tight_layout()


#p4: state transition probability
plt.figure(f'p4', dpi=figure_resolution_in_dpi)
p4 = sns.heatmap(state_markov, annot=True)
p4.set_title(f'state transition probability')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.tight_layout()


#p5: displacement histogram
plt.figure(f'p5', dpi=figure_resolution_in_dpi)
p5 = sns.histplot(data=analysis_data2, x='displacements', stat='percent', hue='state', bins=number_of_bins, kde=True)
p5.set_title(f'displacement histogram')
p5.set_xlabel(r'displacment($\mu m$)')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()


#p6: trajectory length(sec) histogram
plt.figure(f'p6', dpi=figure_resolution_in_dpi)
p6 = sns.histplot(data=analysis_data1, x='duration', stat='percent', hue='state', bins=number_of_bins, kde=True)
p6.set_title(f'trajectory length histogram')
p6.set_xlabel(r'duration($s$)')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.tight_layout()


#p7: MSD
plt.figure(f'p7', dpi=figure_resolution_in_dpi)
p7 = sns.lineplot(data=msd, x=msd['time'], y=msd['mean'], hue='state')
p7.set_title(f'MSD')
p7.set_xlabel(r'time($s$)')
p7.set_ylabel(r'$\frac{\text{MSD}}{\text{2} \cdot \text{dimension}}$ ($\mu m^2$)')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
for state_idx, state in enumerate(states):
    # lower, upper bound related to the number of data (TODO: testing)
    msd_per_state = msd[msd['state'] == state].sort_values('time')
    mus = msd_per_state['mean']
    sigmas = msd_per_state['std']
    lower_bound = [mu - sigma for mu, sigma in zip(mus, sigmas)]
    upper_bound = [mu + sigma for mu, sigma in zip(mus, sigmas)]
    #plt.fill_between(msd_per_state['time'], lower_bound, upper_bound, alpha=.3, color=f'C{state_idx}')
plt.xticks(rotation=90)
plt.tight_layout()


#p8: Ensemble-averaged TAMSD
plt.figure(f'p8', dpi=figure_resolution_in_dpi)
p8 = sns.lineplot(data=tamsd, x=tamsd['time'], y=tamsd['mean'], hue='state')
p8.set_title(f'Ensemble-averaged TAMSD')
p8.set_xlabel(r'lag time($s$)')
p8.set_ylabel(r'$\frac{\text{TAMSD}}{\text{2} \cdot \text{dimension}}$ ($\mu m^2$)')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
for state_idx, state in enumerate(states):
    # lower, upper bound related to the number of data (TODO: testing)
    tamsd_per_state = tamsd[tamsd['state'] == state].sort_values('time')
    mus = tamsd_per_state['mean']
    sigmas = tamsd_per_state['std']
    lower_bound = [mu - sigma for mu, sigma in zip(mus, sigmas)]
    upper_bound = [mu + sigma for mu, sigma in zip(mus, sigmas)]
    #plt.fill_between(tamsd_per_state['time'], lower_bound, upper_bound, alpha=.3, color=f'C{state_idx}')
plt.xticks(rotation=90)
plt.tight_layout()


#p9: bootstrapped distribution with kde(kernel density estimation) plot for averaged mean jump-distances grouped by state.
plt.figure(f'p9', dpi=figure_resolution_in_dpi)
bootstrapped_data = {'averaged_mean_jump_distances':[], 'state':[]}
bootstrapped_results = []
for st in analysis_data1['state'].unique():
    bts = bootstrap([np.array(analysis_data1[analysis_data1['state'] == st]['mean_jump_d'])], np.mean, n_resamples=1000, confidence_level=0.95)
    bootstrapped_data['averaged_mean_jump_distances'].extend(bts.bootstrap_distribution)
    bootstrapped_data['state'].extend([st] * len(bts.bootstrap_distribution))
    bootstrapped_results.append(bts)
p9 = sns.histplot(bootstrapped_data, x=f'averaged_mean_jump_distances', stat='percent', hue='state', bins=bootstrap_bins, kde=False)
p9.set_xlabel(r'bootstrapped mean jump-distances($\mu m$)')
p9.set_title(f'bootstrapped mean jump-distances for each state')
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()


#p10: population of each state as pie chart.
plt.figure(f'p10', dpi=figure_resolution_in_dpi)
state_population = []
state_labels = []
for st in analysis_data1['state'].unique():
    state_population.append(len(analysis_data1[analysis_data1['state'] == st]))
    state_labels.append(st)
plt.pie(x=state_population, labels=state_labels, autopct='%.0f%%')
plt.title('Population of each state')
plt.tight_layout()


plt.show()
