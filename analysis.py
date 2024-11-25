import seaborn as sns
import matplotlib.pyplot as plt
from modules.preprocessing import preprocessing


PIXELMICRONS = 0.16
FRAMERATE = 0.01
CUTOFF = 5
FOLDER = 'condition1'


# Read data from FOLDER and preprocess the data.
analysis_data1, analysis_data2, state_markov, state_graph = preprocessing(folder=FOLDER, pixelmicrons=PIXELMICRONS, framerate=FRAMERATE, cutoff=CUTOFF)


"""
From here, plot functions.
Data is stored in 
1.analysis_data1(DataFrame: contains data of mean_jump_distance, K, alpha, state, length, traj_id)
2.analysis_data2(DataFrame: contains data of displacments, state)
3.state_markov(matrix: contains transition probability)
4.state_graph(network: built from transitions between states(weight: nb of occurence of transitions))
"""
print(f'\nanalysis_data1:\n', analysis_data1)
print(f'\nanalysis_data2:\n', analysis_data2)

#p1: kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f'p1')
p1 = sns.kdeplot(analysis_data1, x=f'mean_jump_d', hue='state')
plt.xlabel(f'mean_jump_distance for each state')
p1.set_title(f'mean_jump_distance')


#p2: joint distribution plot(kde) of alpha(x-axis) and K(y-axis) for each state
p2 = sns.jointplot(data=analysis_data1, x=f"alpha", y=f"K", kind='kde', hue='state')
plt.xlabel(f'alpha')
plt.ylabel(f'K')
p2.fig.suptitle(f'alpha, K distribution for each state')

#p3: histogram of states
plt.figure(f'p3')
p3 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='state')
p3.set_title(f'population of states')


#p4: state transition probability
plt.figure(f'p4')
p4 = sns.heatmap(state_markov, annot=True)
p4.set_title(f'state transition probability')


#p5: displacement histogram
plt.figure(f'p5')
p5 = sns.histplot(data=analysis_data2, x='displacements', stat='percent', hue='state', bins=100)
p5.set_title(f'displacement histogram')
plt.show()

