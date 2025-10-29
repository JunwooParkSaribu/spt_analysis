import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from module.visuailzation import trajectory_visualization, draw_labeled_multigraph
from module.preprocessing import preprocessing, count_cumul_trajs_with_roi, check_roi_passing_traces
from module.preprocessing import preprocessing, inv_cdf_cauchy, cdf_cauchy_2mixture, pdf_cauchy_2mixture,\
    func_to_minimise, cdf_cauchy_1mixture, pdf_cauchy_1mixture, cauchy_location
from module.fileIO.DataLoad import read_multiple_csv, read_multiple_h5s
from scipy.stats import bootstrap, ks_2samp, ecdf
from scipy.optimize import curve_fit
from scipy.optimize import minimize


"""
Major parameters.
"""
FOLDER = f'condition3'  # The folder containing .h5(BI-ADD) or .csv(FreeTrace) files.
PIXELMICRONS = 0.16  # Length of pixel in micrometer. (0.16 -> the length of each pixel is 0.16 micrometer, it varies depending on microscopy.)
FRAMERATE = 0.01  # Exposure time (frame rate) of video for each frame in seconds. (0.01 corresponds to the 10ms) 
CUTOFF = [3, 99999]   # Mininum and maximum length (nb of coordinates) of trajectory to consider
STATE_TO_PLOT = 0  # State number to plot TAMSD and the Cauchy fitting on ratio distribution.
original_data = read_multiple_h5s(path=FOLDER)  # Read BI-ADD results
#original_data = read_multiple_csv(path=FOLDER)   # Read FreeTrace results


"""
Minor parameters.
"""
traj_img_resolution = 80  # Resolution factor of trajectory image. Too high value will exceeding your available space of RAM, resulting the process-kill.
number_of_bins = 50   # Below are the general settings of result plots, you can change here or directly for each plot.
bootstrap_bins = 300
figure_resolution_in_dpi = 200
figure_font_size = 20
y_lim_for_percent = [0, 35]
x_lim_for_mean_jump_distances = [0, 5]
color_palette = ['red','cyan','green','blue','gray','pink']  # Colors for each state of trajectory type



"""
preprocessing generates 9 types of data.
@params: data folder path, pixel microns, frame rate, cutoff
@output: DataFrame, DataFrame, DataFrame, ndarray, networkx grpah, DataFrame, DataFrame, list, dict

preprocessing includes below steps.
1. exclude the trajectory where length is shorter than CUTOFF.
2. convert from pixel unit to micrometer unit with PIXELMICRONS and FRAMERATE.
3. generate 5 DataFrames, 1 ndarray representation of markovchain, 1 graph respresentation of markovchain, 1 list containing states, 1 dictionary containing the duration of transitioning trajectories.
If you want to calculate tamsd, set it as True. It is off in default since tamsd take time to calculate it.
"""
analysis_data1, analysis_data2, analysis_data3, analysis_data4, \
    analysis_data5, state_markov, state_graph, msd, tamsd, states, state_changing_duration = preprocessing(data=original_data,
                                                                                                           pixelmicrons=PIXELMICRONS,
                                                                                                           framerate=FRAMERATE,
                                                                                                           cutoff=CUTOFF,
                                                                                                           selected_state=STATE_TO_PLOT,
                                                                                                           tamsd_calcul=True,
                                                                                                           color_palette=color_palette)
trajectory_image, legend_patch, cmap_for_graph, cmap_for_plot = trajectory_visualization(original_data,
                                                                                         analysis_data1,
                                                                                         CUTOFF,
                                                                                         PIXELMICRONS,
                                                                                         resolution_multiplier=traj_img_resolution, 
                                                                                         roi='', 
                                                                                         scalebar=True,
                                                                                         arrow=False,
                                                                                         color_for_roi=False)


"""
From here, we treat data to make plots or print results.
Data is stored as
1. analysis_data1: (DataFrame: contains data of mean_jump_distance, log10_K, alpha, state, duration, traj_id)
2. analysis_data2: (DataFrame: contains data of displacments, state)
3. analysis_data3: (DataFrame: contains data of angles, state, condition)
4. analysis_data4: (Nested list: contains data of 1d displacements for time lags)
5. analysis_data5: (DataFrame: contains data of ratios, state, condition)
6. state_markov: (matrix: contains transition probability)
7. state_graph: (network: built from transitions between states(weight: nb of occurence of transitions))
8. msd: (DataFrame: contains msd for each state.) 
9. tamsd: (DataFrame: contains ensemble-averaged, time-averaged squared displacement for each state.) 
10. states: classified states beforehand with BI-ADD or other tools.
11. state_changing_duration: list containing the durations of state transitioning trajectories.

Units: 
K: generalized diffusion coefficient, (um^2 / sec^2H) or (px^2 / frame^2H).
H: Hurst exponent (= 2*anomalous diffusion exponent), real number between 0 and 1, exclusive.
mean_jump_disatnce: set of averages of jump distances in um.
state: states of trajectories, re-ordered from slow to fast.
duration: duration(length) of trajectory in seconds.
"""
print(f"\nanalysis_data1:\n", analysis_data1)



#p1: histogram with kde(kernel density estimation) plot of mean jump distance grouped by state.
plt.figure(f"p1", dpi=figure_resolution_in_dpi)
p1 = sns.histplot(analysis_data1, x=f"mean_jump_d", stat='percent', hue='state', bins=number_of_bins, palette=cmap_for_plot, kde=True)
p1.set_xlabel(r"mean jump-distance($\mu m$)")
p1.set_title(f"Mean jump-distances for {FRAMERATE} sec")
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()


#p2: H - K distribution plot of H(x-axis) and K(y-axis) for each state
fig, axs = plt.subplots(1, 1, layout='constrained', dpi=figure_resolution_in_dpi, num=f'p2')
colormap = sns.color_palette("mako", as_cmap=True)
axs.add_patch(Rectangle((0, -100), 1.0, 1000, ec='none', fc=colormap(0), zorder=0))
sns.kdeplot(
    data=analysis_data1, x="H", y="K", fill=True, ax=axs, thresh=0, levels=100, cmap=colormap, log_scale=(False, True), bw_adjust=1.0,
)
axs.set_yscale('log')
axs.set_ylabel(f"K (generalised diffusion coefficient)")
axs.set_xlabel(f"H (Hurst exponent)")
fig.suptitle(f"Cluster of estimated H and K of individual trajectories for {FRAMERATE}sec.")
axs.set_xlim([0.0, 1.0])
axs.set_ylim([10**-3, (10**1)*2])
axs.set(xlabel=None, ylabel=None)
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)


#p3: histogram of states
plt.figure(f"p3", dpi=figure_resolution_in_dpi)
p3 = sns.histplot(data=analysis_data1, x="state", stat='percent', hue='state', palette=cmap_for_plot)
p3.set_title(f"Population of states")
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.tight_layout()



#p4: state transitioning probabilities
"""
Self-loop indicates the number of non state-changing trajectories in the graph.
"""
if len(states) >= 2:  # make plot only when the total number of different states is >= 2.
    print("\n** Making figures... please wait, it takes time if the number of trajectories is big. **\n")
    fig, axs = plt.subplots(nrows=2, ncols=len(states), num=f'p4', dpi=figure_resolution_in_dpi)
    #duration_bins = np.linspace(0, 10, 100)  # bin range: duration in seconds
    for st, ax in zip(states, axs[0]):
        for next_st in states:
            if st != next_st:
                sns.histplot(state_changing_duration[tuple([st, next_st])], kde=True, ax=ax, label=f'{st} -> {next_st}')
                ax.set_title(f'Duration of transitioning trajectories for the state: {st}')
                ax.set_xlabel(r'Duration (sec)')
                ax.legend()
    draw_labeled_multigraph(G=state_graph, attr_names=["count", "freq"], cmap=cmap_for_graph, ax=axs[1, 0])
    axs[1, 1].imshow(trajectory_image)
    axs[1, 1].legend(handles=legend_patch, loc='upper right', borderaxespad=0.)
    fig.tight_layout()



#p5: displacement histogram
plt.figure(num=f"p5", dpi=figure_resolution_in_dpi)
p5 = sns.histplot(data=analysis_data2, x='2d_displacement', stat='percent', hue='state', bins=number_of_bins, kde=True, palette=cmap_for_plot)
p5.set_title(f"Displacements for {FRAMERATE} sec")
p5.set_xlabel(r"displacment($\mu m$)")
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()



#p6: trajectory length(sec) histogram
plt.figure(num=f"p6", dpi=figure_resolution_in_dpi)
p6 = sns.histplot(data=analysis_data1, x='duration', stat='percent', hue='state', binwidth=FRAMERATE, kde=True, palette=cmap_for_plot)
p6.set_title(f"Trajectory durations")
p6.set_xlabel(r"duration($s$)")
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.tight_layout()



#p7: Ensemble-averaged squared displacement
plt.figure(f"p7", dpi=figure_resolution_in_dpi)
p7 = sns.lineplot(data=msd, x=msd['time'], y=msd['mean'], hue='state', palette=cmap_for_plot)
p7.set_title(f"MSD")
p7.set_xlabel(r"time($s$)")
p7.set_ylabel(r"$\frac{\text{MSD}}{\text{2} \cdot \text{dimension}}$ ($\mu m^2$)")
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



#p9: Bootstrapped distribution with kde(kernel density estimation) plot for averaged mean jump-distances grouped by state.
plt.figure(f"p9", dpi=figure_resolution_in_dpi)
bootstrapped_data = {'averaged_mean_jump_distances':[], 'state':[]}
bootstrapped_results = []
for st in analysis_data1['state'].unique():
    bts = bootstrap([np.array(analysis_data1[analysis_data1['state'] == st]['mean_jump_d'])], np.mean, n_resamples=1000, confidence_level=0.95)
    bootstrapped_data['averaged_mean_jump_distances'].extend(bts.bootstrap_distribution)
    bootstrapped_data['state'].extend([st] * len(bts.bootstrap_distribution))
    bootstrapped_results.append(bts)
p9 = sns.histplot(bootstrapped_data, x=f"averaged_mean_jump_distances", stat='percent', hue='state', bins=bootstrap_bins, kde=False, palette=cmap_for_plot)
p9.set_xlabel(r"bootstrapped mean jump-distances($\mu m$)")
p9.set_title(f"Bootstrapped mean jump-distances for each state")
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.xticks(rotation=90)
plt.ylim(y_lim_for_percent)
plt.xlim(x_lim_for_mean_jump_distances)
plt.tight_layout()



#p10: Population of each state as pie chart.
plt.figure(f"p10", dpi=figure_resolution_in_dpi)
state_population = []
state_labels = []
state_colors = []
for st in analysis_data1['state'].unique():
    state_population.append(len(analysis_data1[analysis_data1['state'] == st]))
    state_labels.append(st)
    for st_c in analysis_data1[analysis_data1['state'] == st]['color'].unique():
        if st_c != 'yellow':
            state_colors.append(st_c)
plt.pie(x=state_population, labels=[f'Nb of state {state_labels[st]}: {state_population[st]}' for st in state_labels], autopct='%.0f%%', colors=state_colors)
plt.title(f"Population of each state")
plt.tight_layout()



#p11: Trajectory image with respect to each state.
plt.figure(f"p11", dpi=figure_resolution_in_dpi)
plt.imshow(trajectory_image)
plt.legend(handles=legend_patch, loc='upper right', borderaxespad=0.)
plt.xticks([])
plt.yticks([])
plt.tight_layout()



#p12: Accumulated number of trajectories in ROI or in the entire video. If you have ROI file, please fill the roi_file parameter.
start_frame = 1  # number of start frame to accumulate the observed trajectories
end_frame = 100  # number of end frame.
fig, axs = plt.subplots(nrows=2, ncols=1, num=f"p12")
traj_counts, acc_traj_counts = count_cumul_trajs_with_roi(original_data, roi_file=None, start_frame=start_frame, end_frame=end_frame, cutoff=CUTOFF)
x_vlines = []
x_axis = []
for idx in range(len(traj_counts)):
    count = traj_counts[idx]
    x_axis.append((idx + 1) * FRAMERATE)
    x_vlines.extend([(idx + 1) * FRAMERATE] * count)
axs[0].plot(x_axis, acc_traj_counts, c='black')
axs[1].vlines(x_vlines, 0, 1, colors='black')
axs[0].set_xlim([0, x_axis[-1] + FRAMERATE])
axs[1].set_xlim([0, x_axis[-1] + FRAMERATE])
axs[0].set_ylabel(r"Accumulated counts")
fig.supxlabel(r"Time (sec)")
fig.suptitle(f"Number of accumulated trajectories from {round(start_frame*FRAMERATE,2)} to {round(end_frame*FRAMERATE,2)} sec.")
plt.tight_layout()



#p13: Angle histogram
fig, axs = plt.subplots(1, 2, num=f"p13", figsize=(18, 9))
sns.histplot(data=analysis_data3, x='angle', stat='proportion', hue='state', common_norm=False, bins=number_of_bins, kde=True, ax=axs[0], kde_kws={'bw_adjust': 1}, palette=cmap_for_plot)
sns.ecdfplot(data=analysis_data3, x='angle', stat='proportion', hue='state', ax=axs[1], palette=cmap_for_plot)
axs[0].set_title(f"angle histogram")
axs[0].set_xlabel(r"Angle (degree)")
axs[1].set_title(f"angle CDF")
axs[1].set_xlabel(r"Angle (degree)")
cmap = mpl.colormaps['cividis']
custom_lines = [Line2D([0], [0], color=cmap(i/(len(states) - 1)), lw=2) for i in range(len(states) - 1)]
legend_labels = []
legend_results = []
for idx in range(1, len(states)):
    gt = analysis_data3[analysis_data3['state']==states[0]]['angle']
    comp = analysis_data3[analysis_data3['state']==states[idx]]['angle']
    ecdf_comp = ecdf(comp)
    ecdf_gt = ecdf(gt)
    result = ks_2samp(gt, comp, method='exact')
    axs[1].vlines(result.statistic_location, ecdf_comp.cdf.evaluate(result.statistic_location), ecdf_gt.cdf.evaluate(result.statistic_location), colors=cmap((idx-1)/(len(states) - 1)), alpha=0.6)
    legend_labels.append(f"D: {np.round(result.statistic, 3)}, state:{states[0]} vs state:{states[idx]}")
    legend_results.append(np.round(result.statistic, 3))
custom_lines = np.array(custom_lines)[np.argsort(legend_results)]
legend_labels = np.array(legend_labels)[np.argsort(legend_results)]
old_legend = axs[0].legend_
handles = old_legend.legend_handles
labels = [t.get_text() for t in old_legend.get_texts()]
axs[0].legend(handles, labels)
axs[1].legend(custom_lines, legend_labels, title='KS test')
plt.tight_layout()



#p14: 1D ratio distribution with Cauchy fitting for the selecetd state.
plt.figure(num=f"p14", dpi=figure_resolution_in_dpi)
vline_ymax = 0.018
hist_bins = np.arange(-10000, 10000, 0.05)
target = analysis_data5[analysis_data5['state']==STATE_TO_PLOT]['1d_ratio'].to_numpy()
hist, bin_edges = np.histogram(target, bins=hist_bins, density=True)
plt.hist(bin_edges[:-1], bin_edges, weights=hist / np.sum(hist), alpha=1.0, histtype='stepfilled', zorder=0, linewidth=2, color=color_palette[STATE_TO_PLOT])
xs = hist_bins[:-1] + (hist_bins[1] - hist_bins[0])/2
cons = ({'type': 'ineq', 'fun': lambda k:  k[2] - 0.2},
        #{'type': 'eq', 'fun': lambda k:  k[0] + k[1] + k[2] - 1},
        )
res1 = minimize(func_to_minimise, x0=[0.5, 0.1], args=(pdf_cauchy_1mixture, xs, hist / np.sum(hist)), 
                method='trust-constr',
                #constraints=cons, 
                bounds=((1e-6, 0.9999), (1e-6, None),),
                tol=1e-10,
                )
params, residual = res1.x, res1.fun
plt.vlines(cauchy_location(params[0]), ymin=0, ymax=vline_ymax, colors='red', alpha=0.8, zorder=5, linewidth=5)
plt.plot(xs, pdf_cauchy_1mixture(xs, *params), c='red',
            label=r"$\hat{H}$: %5.4f" % params[0], alpha=0.9, zorder=3, linewidth=5)
plt.xlim([-5, 5])
plt.ylim([0, vline_ymax])
plt.yticks(fontsize=figure_font_size)
plt.xticks(fontsize=figure_font_size)
plt.legend(title=r"H", fontsize="40", title_fontsize="40", loc='upper right')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()



#p15: Ensemble-averaged, time-averaged squared displacement for the selected state.
if tamsd is not None:
    fig, axs = plt.subplots(num=f"p15", dpi=figure_resolution_in_dpi)
    curve = lambda K, H, tau:K*tau**(2*H)
    def Gauss(x, A, var):
        return A * np.exp(-1/2. * (x**2/var))
    bins_for_disps = np.arange(-20, 20, 0.25)
    bins_for_ratios = np.arange(-10000, 10000, 0.05)
    tamsd = tamsd[tamsd['state'] == STATE_TO_PLOT]
    tamsd = tamsd.reset_index(drop=True)
    tamsd = tamsd.dropna()

    sns.lineplot(data=tamsd[tamsd['state'] == STATE_TO_PLOT], x='time', y='mean', ax=axs, c='black', label='Empirical MSD')
    disps_for_prog = analysis_data4[STATE_TO_PLOT]
    estimed_vars = []
    for lag in range(len(disps_for_prog)):
        disp_fit = disps_for_prog[lag]
        if len(disp_fit) > 2:
            disp_fit_y = np.histogram(disp_fit, bins=bins_for_disps, density=True)[0]
            try:
                parameters, _ = curve_fit(Gauss, bins_for_disps[:-1]+0.05, disp_fit_y)
                fit_A, fit_B = parameters
                fit_y = Gauss(bins_for_disps, fit_A, fit_B)
                estimated_var = fit_B / 2
                estimed_vars.append(estimated_var)
            except:
                estimed_vars.append(-1)
        else:
            estimed_vars.append(-1)
    estimed_vars = np.array(estimed_vars)
    estimed_vars[0] = 0
    estimed_vars = estimed_vars[estimed_vars > -0.0001]

    target = analysis_data5[analysis_data5['state']==STATE_TO_PLOT]['1d_ratio'].to_numpy()
    hist, bin_edges = np.histogram(target, bins=bins_for_ratios, density=True)
    xs = bin_edges[1:] - (bin_edges[1] - bin_edges[0])/2
    res1 = minimize(func_to_minimise, x0=[0.5, 0.1], args=(pdf_cauchy_1mixture, xs, hist / np.sum(hist)), 
                    method='trust-constr',
                    bounds=((1e-6, 0.9999), (1e-6, None),),
                    tol=1e-10,
                    )
    params, residual = res1.x, res1.fun
    estimed_h = params[0]

    print(f"\nEstimated K with gaussian fitting: {round(estimed_vars[1], 5)},\n\
          Estimated H: {round(estimed_h, 5)},\n\
          Residual: {round(residual, 5)},\n\
          Estimated K for {FRAMERATE}sec with squared displacement:{tamsd['mean'].to_numpy()[1]},\n\
          Estimated K for 1sec: {round(tamsd['mean'].to_numpy()[1] * (1/(FRAMERATE**(2*estimed_h))), 3)}, Unit:um^2/sec^{2*round(estimed_h, 3)}")
    axs.plot(tamsd['time'], curve(tamsd['mean'][1], estimed_h, tamsd['time'].to_numpy() / FRAMERATE), label=f'Estimated Curve: K and H\n({round(tamsd['mean'].to_numpy()[1], 4)}, {round(np.mean(estimed_h), 4)})', c=color_palette[STATE_TO_PLOT])
    axs.set_xlabel(f"Time lag")
    axs.set_ylabel(f"Squared displacement")
    fig.suptitle(f"MSD for the state {STATE_TO_PLOT}. Empirical and estimated evolution of molecular diffusion over time")
    plt.yticks(fontsize=figure_font_size)
    plt.xticks(fontsize=figure_font_size)
    plt.legend()
    #plt.xticks(rotation=90)
    #plt.xlim([-0.1, 60])
    #plt.ylim([-0.1, 40])
    plt.tight_layout()


plt.show()
