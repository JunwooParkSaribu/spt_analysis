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
import matplotlib.ticker as ticker




"""
Minor parameters.
"""
traj_img_resolution = 80  # Resolution factor of trajectory image. Too high value will exceeding your available space of RAM, resulting the process-kill.
number_of_bins = 50   # Below are the general settings of result plots, you can change here or directly for each plot.
bootstrap_bins = 300
figure_resolution_in_dpi = 300
figure_font_size = 20
y_lim_for_percent = [0, 35]
x_lim_for_mean_jump_distances = [0, 5]
color_palette = ['red','cyan','green','blue','gray','pink']  # Colors for each state of trajectory type


plt.rc('xtick', labelsize=figure_font_size) 
plt.rc('ytick', labelsize=figure_font_size) 

nb_traj = 5000
noise = 0.0
theoretical_H = 0.5
theoretical_K = 1.0

PIXELMICRONS = 1  # Length of pixel in micrometer. (0.16 -> the length of each pixel is 0.16 micrometer, it varies depending on microscopy.)
FRAMERATE = 1  # Exposure time (frame rate) of video for each frame in seconds. (0.01 corresponds to the 10ms) 
CUTOFF = [3, 99999]   # Mininum and maximum length (nb of coordinates) of trajectory to consider
STATE_TO_PLOT = 0  # State number to plot TAMSD and the Cauchy fitting on ratio distribution.

total_times = 100
total_exps = np.arange(0, 100, 1)



lengths = [[] for _ in range(total_times)]
empirical_msds = [[] for _ in range(total_times)]
empirical_stds = [[] for _ in range(total_times)]
gaussian_pdf = [[] for _ in range(total_times)]
gaussian_cdf = [[] for _ in range(total_times)]
emp_pdf = [[] for _ in range(total_times)]
emp_cdf = [[] for _ in range(total_times)]
for exp_number in total_exps:
    print(f"EXP NUMBER: {exp_number} / {total_exps}")
    FOLDER = f'fbm_simulations/{nb_traj}_{noise}_{theoretical_H}/{exp_number}'  # The folder containing .h5(BI-ADD) or .csv(FreeTrace) files.
    original_data = read_multiple_csv(path=FOLDER)   # Read FreeTrace results

    analysis_data1, analysis_data2, analysis_data3, analysis_data4, \
        analysis_data5, state_markov, state_graph, msd, tamsd, states, state_changing_duration = preprocessing(data=original_data,
                                                                                                            pixelmicrons=PIXELMICRONS,
                                                                                                            framerate=FRAMERATE,
                                                                                                            cutoff=CUTOFF,
                                                                                                            selected_state=STATE_TO_PLOT,
                                                                                                            tamsd_calcul=True,
                                                                                                            color_palette=color_palette)


    times = np.arange(total_times)#tamsd['time'].to_numpy()
    length = analysis_data1['duration'].to_numpy()

    
    length_hist, length_bin_edges = np.histogram(length, bins=times, density=False)

    for time_lag in times[:-1]:
        #empirical_msds[time_lag].append(emp_msd[time_lag])
        lengths[time_lag].append(length_hist[time_lag])

    #p14: 1D ratio distribution with Cauchy fitting for the selecetd state.
    vline_ymax = 0.018
    hist_bins = np.arange(-10000, 10000, 0.1)
    target = analysis_data5[analysis_data5['state']==STATE_TO_PLOT]['1d_ratio'].to_numpy()
    hist, bin_edges = np.histogram(target, bins=hist_bins, density=True)

    xs = hist_bins[:-1] + (hist_bins[1] - hist_bins[0])/2
    cons = ({'type': 'ineq', 'fun': lambda k:  k[2] - 0.2},
            #{'type': 'eq', 'fun': lambda k:  k[0] + k[1] + k[2] - 1},
            )
    target = np.sort(target)

    res1 = minimize(func_to_minimise, x0=[0.5, 0.1], args=(cdf_cauchy_1mixture, target, np.cumsum(np.array([1] * len(target)) / len(target))), 
                    #method='trust-constr',
                    #constraints=cons, 
                    bounds=((1e-6, 0.9999), (1e-6, None),),
                    tol=1e-10,
                    )
    params, residual = res1.x, res1.fun
    cdf_cacuhy_val = params[0]



    #p15: Ensemble-averaged, time-averaged squared displacement for the selected state.
    if tamsd is not None:
        curve = lambda K, H, tau:K*tau**(2*H)
        def Gauss(x, A, var):
            return A * np.exp(-1/2. * (x**2/var))
        bins_for_disps = np.arange(-20, 20, 0.25)
        bins_for_ratios = np.arange(-10000, 10000, 0.05)
        tamsd = tamsd[tamsd['state'] == STATE_TO_PLOT]
        tamsd = tamsd.reset_index(drop=True)
        #tamsd = tamsd.dropna()
        #print(tamsd)
        emp_msd = tamsd[tamsd['state'] == STATE_TO_PLOT]['mean'].to_numpy()
        emp_std = tamsd[tamsd['state'] == STATE_TO_PLOT]['std'].to_numpy()
        for time_lag in range(len(emp_msd)):
            empirical_msds[time_lag].append(emp_msd[time_lag])
            empirical_stds[time_lag].append(emp_std[time_lag])

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


        e_pdf = curve(tamsd['mean'][1], estimed_h, times / FRAMERATE)
        g_pdf = curve(estimed_vars[1], estimed_h, times / FRAMERATE)
        e_cdf = curve(tamsd['mean'][1], cdf_cacuhy_val, times / FRAMERATE)
        g_cdf = curve(estimed_vars[1], cdf_cacuhy_val, times / FRAMERATE)

        for time_lag in times:
            gaussian_pdf[time_lag].append(g_pdf[time_lag])
            gaussian_cdf[time_lag].append(g_cdf[time_lag])
            emp_pdf[time_lag].append(e_pdf[time_lag])
            emp_cdf[time_lag].append(e_cdf[time_lag])
        




for idx in range(len(gaussian_pdf)):
    if len(gaussian_pdf[idx]) == 0:
        gaussian_pdf[idx].append(0)
    if len(gaussian_cdf[idx]) == 0:
        gaussian_cdf[idx].append(0)
    if len(emp_pdf[idx]) == 0:
        emp_pdf[idx].append(0)
    if len(emp_cdf[idx]) == 0:
        emp_cdf[idx].append(0)
    if len(empirical_msds[idx]) == 0:
        empirical_msds[idx].append(0)
    if len(empirical_stds[idx]) == 0:
        empirical_stds[idx].append(0)
    if len(lengths[idx]) == 0:
        lengths[idx].append(0)
lengths = lengths[2:]


emp_msd_avgs = [np.mean(vals) for vals in empirical_msds]
emp_pdf_avgs = [np.mean(vals) for vals in emp_pdf]
emp_cdf_avgs = [np.mean(vals) for vals in emp_cdf]
gauss_pdf_avgs = [np.mean(vals) for vals in gaussian_pdf]
gauss_cdf_avgs = [np.mean(vals) for vals in gaussian_cdf]
length_avgs = [np.mean(vals) for vals in lengths]

emp_msd_stds = [np.std(vals) for vals in empirical_msds]
emp_pdf_stds = [np.std(vals) for vals in emp_pdf]
emp_cdf_stds = [np.std(vals) for vals in emp_cdf]
gauss_pdf_stds = [np.std(vals) for vals in gaussian_pdf]
gauss_cdf_stds = [np.std(vals) for vals in gaussian_cdf]
length_stds = [np.std(vals) for vals in lengths]

fig, axs = plt.subplots(nrows=2, ncols=1, dpi=figure_resolution_in_dpi)
axs[0].plot(np.arange(0, total_times, 1), curve(theoretical_K, theoretical_H, np.arange(0, total_times, 1) / FRAMERATE), c='green', zorder=0)
#axs[0].plot(np.arange(0, total_times, 1), emp_msd_avgs, c='black')
#axs[0].plot(np.arange(0, total_times, 1), emp_pdf_avgs, c='red')
#axs[0].plot(np.arange(0, total_times, 1), emp_cdf_avgs, c='blue')
#axs[0].plot(np.arange(0, total_times, 1), gauss_pdf_avgs, c='purple')
#axs[0].plot(np.arange(0, total_times, 1), gauss_cdf_avgs, c='orange')
#axs[0].boxplot(empirical_msds, positions=np.arange(0, total_times, 1))
#axs[0].boxplot(emp_cdf, positions=np.arange(0, total_times, 1))
#axs[1].boxplot(lengths, positions=np.arange(0, total_times, 1))
axs[0].errorbar(np.arange(0, total_times, 1), emp_msd_avgs, emp_msd_stds, linestyle='None', marker='X', capsize=5, c='black', alpha=0.8, ms=4, zorder=1)
#axs[0].errorbar(np.arange(0, total_times, 1), emp_pdf_avgs, emp_pdf_stds, linestyle='None', marker='o', capsize=5, c='blue', alpha=0.7, ms=4)
axs[0].errorbar(np.arange(0, total_times, 1), emp_cdf_avgs, emp_cdf_stds, linestyle='None', marker='P', capsize=5, c='red', alpha=0.7, ms=4, zorder=20)

axs[1].errorbar(np.arange(3, len(length_avgs)+3, 1), length_avgs, length_stds, linestyle='None', marker='X', capsize=5, c='black', alpha=1.0, ms=4)
axs[0].set_xlim([-0.5, 18.5])
axs[1].set_xlim([-0.5, 18.5])
axs[0].set_ylim([-0.5, np.max(emp_cdf_avgs[:23]) + np.max(emp_cdf_stds[:23]) + 1])
#axs[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
#axs[1].xaxis.set_major_locator(ticker.MultipleLocator(5))


"""
for plotss, color in zip([empirical_msds, emp_pdf], ['black', 'red']):
    violin = axs[0].violinplot(plotss, positions=np.arange(0, total_times, 1),
                               showmeans=True,
                               showmedians=False)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(1)
"""
plt.savefig(f"fbm_simulations/{nb_traj}_{noise}_{theoretical_H}.png", transparent=True)
#plt.show()
