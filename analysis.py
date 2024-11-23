import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from modules.fileIO import DataLoad
from scipy import stats
import scipy


def read_files_by_each_time(path: str, plotList: list, cutoff=2) -> dict:
    """
    Read histone files and return the stored H2B object by each time.
    """
    histones = {}
    print("------ Number of trajectory for each time ------")
    for time in plotList:
        histones[time] = []
        h = DataLoad.read_files([f'{path}/{time}'], cutoff=cutoff, chunk=False)[0]
        for h2b in h:
            histones[time].append(h[h2b])
        print(f'{time}: {len(histones[time])}\n')
    return histones


def read_ratio_file(file):
    ratio = []
    with open(file) as f:
        line = f.readline()
        str_ratio = line.split(':')[-1].split('(')[-1].split(')')[0].strip().split(',')
        try:
            for r in str_ratio:
                ratio.append(float(r.strip()))
        except Exception as e:
            print(e)
            print('Err while reading ratio files')
    return ratio


def read_coef_file(file):
    coefs = {}
    coef = []
    with open(file) as f:
        lines = f.readlines()
        old_filename = lines[1].strip().split(',')[0].strip()
        old_h2b_id = lines[1].strip().split(',')[1].strip()
        for line in lines[1:]:  # first line is fieldname
            try:
                cur_line = line.strip().split(',')
                if len(cur_line[1]) > 0:          
                    h2b_id = cur_line[1].strip()
                    filename = cur_line[0].strip()
                    if h2b_id != old_h2b_id or old_filename != filename:
                        coefs[(old_filname, old_h2b_id)] = coef
                    coef = []
                val = cur_line[-1].strip()
                coef.append(float(val))
                old_h2b_id = h2b_id
                old_filname = filename
            except Exception as e:
                print(e)
                print('Err while reading diff_coef files')
        coefs[(old_filname, old_h2b_id)] = coef
    return coefs


def read_classficiation_result_file(file):
    results = {}
    with open(file, errors='ignore') as f:
        lines = f.readlines()
        for line in lines[1:]:  # first line is fieldname
            try:
                cur_line = line.strip().split(',')
                filename = cur_line[0].strip()
                h2b_id = cur_line[1].strip()
                h2b_class = int(float(cur_line[2].strip()))
                results[(filename, h2b_id)] = h2b_class
            except Exception as e:
                print(e)
                print('Err while reading classfication result files')
    return results


def dir_search(path):
    ratio_dict = {}
    diffcoef_dict = {}
    classification_dict = {}
    for root, dirs, files in os.walk(path, topdown=False):
        root = root.replace('\\', '/')
        hint = root.strip().split('/')[-1]
        if hint == path.split('/')[-1]:
            continue
        if hint not in ratio_dict:
            ratio_dict[hint] = []
        if hint not in diffcoef_dict:
            diffcoef_dict[hint] = {}
        if hint not in classification_dict:
            classification_dict[hint] = {}

        for file in files:
            if '_ratio.txt' in file:
                ratio = read_ratio_file(f'{root}/{file}')
                ratio_dict[hint].append(ratio)
            if '_diffcoef.csv' in file:
                coef = read_coef_file(f'{root}/{file}')
                diffcoef_dict[hint] = diffcoef_dict[hint] | coef
            if '.trxyt.csv' in file:
                classification_results = read_classficiation_result_file(f'{root}/{file}')
                classification_dict[hint] = classification_dict[hint] | classification_results
    return ratio_dict, diffcoef_dict, classification_dict


def MSD(histones, plotList, fontSize, t_limit):
    """
    Mean Squared Displacement.
    The average displacement of immobile H2B for each time,
    Formula(1) is in the attached image.
    ref: https://iopscience.iop.org/article/10.1088/1367-2630/15/4/045011.
    """
    msd = {}
    x_axis = {}
    ret_msd = {}
    for time in plotList:
        histone_list = histones[time]
        disps = []
        for h2b in histone_list:
            trajectory = h2b.get_trajectory()
            t_seq = h2b.get_time()
            # ref_position and ref_t are the first detected position and time of a molecule
            ref_position = trajectory[0]
            ref_t = t_seq[0]
            displacement = {}
            # MSD calculation
            for pos, t in zip(trajectory, t_seq):
                displacement[np.round(t - ref_t, 5)] = \
                    np.sqrt((pos[0] - ref_position[0])**2 + (pos[1] - ref_position[1])**2) ** 2
            disps.append(displacement)

        # Take all possible time(sec) for the x-axis of plot
        all_possible_times = set()
        for disp in disps:
            tmp = list(disp.keys())
            for tp in tmp:
                all_possible_times.add(tp)
        all_possible_times = list(all_possible_times)
        all_possible_times.sort()
        x_axis[time] = all_possible_times.copy()

        # y_axis values for a given time, if there is a blink, pass to the next.
        msd[time] = {}
        for t in all_possible_times:
            msd[time][t] = []
            for disp in disps:
                if t in disp:
                    msd[time][t].append(disp[t])

    plt.figure(figsize=(8, 8))
    for time, c in zip(plot_list, ['red', 'orange', 'green', 'blue']):
        y_vals = []
        for t in x_axis[time]:
            if t > t_limit:
                break
            if len(msd[time][t]) < 5:
                break
            y_vals.append(np.mean(msd[time][t]))
        plt.plot(np.arange(len(y_vals))/100, y_vals, label=str(time), alpha=0.7, c=c)
        ret_msd[time] = [np.arange(len(y_vals))/100, y_vals]
    plt.ylabel('MSD($um^{2}$)', fontsize=fontSize)
    plt.xlabel('Time(sec)', fontsize=fontSize)
    #plt.ylim(0, 0.15)
    plt.legend()
    return ret_msd


def TAMSD(histones, plotList, fontSize, t_limit):
    """
    Time Averaged Mean Squared Displacement.
    The time averaged displacement of immobile H2B for each time.
    Formula(2) is in the attached image.
    ref: https://iopscience.iop.org/article/10.1088/1367-2630/15/4/045011
    """
    tamsd = dict()
    x_vals_mean = {}
    y_vals_mean = {}
    for time in plotList:
        tamsd[time] = {}
        histone_list = histones[time]
        max_time_gap = -999
        for h2b in histone_list:
            tamsd[time][h2b] = {}
            t = h2b.get_time()
            max_t = t[-1]
            min_t = t[0]
            max_time_gap = max(max_time_gap, np.round(max_t - min_t, 5))

        time_gaps = np.arange(0.00, max_time_gap+0.01, 0.01)
        for h2b in histone_list:
            for delta_t in time_gaps:
                delta_t = np.round(delta_t, 5)
                tamsd[time][h2b][delta_t] = []

        for h2b in histone_list:
            trajectory = h2b.get_trajectory()
            t_seq = h2b.get_time()
            displacement = {}
            for i in range(len(t_seq)):
                for j in range(i, len(t_seq)):
                    t_gap = np.round(t_seq[j] - t_seq[i], 5)
                    disp = (trajectory[j][0] - trajectory[i][0]) ** 2 \
                           + (trajectory[j][1] - trajectory[i][1]) ** 2
                    if t_gap in displacement:
                        displacement[t_gap].append(disp)
                    else:
                        displacement[t_gap] = [disp]

            for t_gap in displacement:
                tamsd[time][h2b][t_gap].append(np.mean(displacement[t_gap]))

        #plt.figure(f'{time}')
        x_vals_mean[time] = time_gaps.copy()
        y_vals_mean[time] = []
        tmp_y_vals_mean = {}
        for h2b in histone_list:
            single_x_vals = []
            single_y_vals = []
            for t_gap in tamsd[time][h2b]:
                if t_gap in tmp_y_vals_mean:
                    tmp_y_vals_mean[t_gap].extend(tamsd[time][h2b][t_gap].copy())
                else:
                    tmp_y_vals_mean[t_gap] = tamsd[time][h2b][t_gap].copy()
                if len(tamsd[time][h2b][t_gap]) > 0:
                    single_x_vals.append(t_gap)
                    y_val = tamsd[time][h2b][t_gap][0]
                    single_y_vals.append(y_val)

            # TAMSD of single particle
            #plt.plot(single_x_vals, single_y_vals, c='red', alpha=0.3)

        for t_gap in tmp_y_vals_mean:
            y_vals_mean[time].append(np.mean(tmp_y_vals_mean[t_gap]))
        
        """
        # Average of particle's TAMSD
        plt.plot(x_vals_mean[time], y_vals_mean[time], c='blue', alpha=0.7)
        blue_line = mlines.Line2D([], [], color='blue', label='Avg')
        red_line = mlines.Line2D([], [], color='red', label='TAMSD of single particle')
        plt.legend(handles=[blue_line, red_line])
        plt.ylabel('TAMSD($um^{2}$)', fontsize=fontSize)
        plt.xlabel('Time(sec)', fontsize=fontSize)
        plt.title(f'{time}')
        """
    
    types = {0:'immobile', 1:'hybrid', 2:'mobile'}
    plt.figure(f'Avgs', figsize=(8, 8))
    for time in plotList:
        plt.plot(x_vals_mean[time], y_vals_mean[time], label=str(time), alpha=0.7)
    plt.ylabel('TAMSD($um^{2}$)', fontsize=fontSize)
    plt.xlabel('Time(sec)', fontsize=fontSize)
    plt.ylim(0, 0.07)
    plt.legend()
    
    for time in plotList:
        x_vals_mean[time] = np.array(x_vals_mean[time])[:int(t_limit*100)]
        y_vals_mean[time] = np.array(y_vals_mean[time])[:int(t_limit*100)]

    return x_vals_mean, y_vals_mean


def box_plots(path, plotList, boxColors, fontSize):
    plotList = plotList.copy()
    # Read classification result files (ratio, diffusion coef)
    ratio, coefs, types = dir_search(path)
    key_list = list(ratio.keys())
    for key in key_list:
        if key not in plotList:
            del ratio[key]

    nb_time = len(plotList)
    data = pd.DataFrame()
    index = 0
    for time in plotList:
        immobiles = np.array(ratio[time])[:, 0]
        for imm in immobiles:
            data = pd.concat([data, pd.DataFrame({'time':[time], 'cum_population':[imm], 'type':['immobile']}, index=[index])])
            index += 1
        hybrids = np.array(ratio[time])[:, 1]
        for hyb in hybrids:
            data = pd.concat([data, pd.DataFrame({'time':[time], 'cum_population':[np.mean(immobiles) + hyb], 'type':['hybrid']}, index=[index])])
            index += 1
        mobiles = np.array(ratio[time])[:, 2]
        for mob in mobiles:
            data = pd.concat([data, pd.DataFrame({'time':[time], 'cum_population':[np.mean(immobiles) + np.mean(hybrids) + mob], 'type':['mobile']}, index=[index])])
            index += 1

    # Poluation of each class
    print('\n########  Population  ########')
    for time in range(0, len(plotList)):
        print(f'---------- {plotList[time]} ---------')
        print(f'immobile -> mean: {np.mean(np.array(ratio[plotList[time]])[:, 0])}\tstd: {np.std(np.array(ratio[plotList[time]])[:, 0])}')
        print(f'hybrid   -> mean: {np.mean(np.array(ratio[plotList[time]])[:, 1])}\tstd: {np.std(np.array(ratio[plotList[time]])[:, 1])}')
        print(f'mobile   -> mean: {np.mean(np.array(ratio[plotList[time]])[:, 2])}\tstd: {np.std(np.array(ratio[plotList[time]])[:, 2])}')
        print(f'------------------------')
        print(f'')

    # T-test between Before and each class
    print('\n########  Population ttest  ########')
    for time in range(1, len(plotList)):
        print(f'result between {plotList[0]} and {plotList[time]}')
        print('Immobile: ',
              stats.ttest_ind(np.array(ratio[plotList[0]])[:, 0], np.array(ratio[plotList[time]])[:, 0]))
        print('Hybrid: ',
              stats.ttest_ind(np.array(ratio[plotList[0]])[:, 1], np.array(ratio[plotList[time]])[:, 1]))
        print('Mobile: ',
              stats.ttest_ind(np.array(ratio[plotList[0]])[:, 2], np.array(ratio[plotList[time]])[:, 2]))
        print()

    plt.figure('h2b types population')
    sns.set(style='white')
    hue_order = ['mobile', 'hybrid', 'immobile']
    palette = {'mobile': 'cyan', 'hybrid': 'green', 'immobile': 'magenta'}
    sns.barplot(data=data, x='time', y='cum_population', hue='type', errorbar='ci', width=0.6,
                hue_order=hue_order, dodge=False, capsize=0.05, palette=palette, alpha=0.85, legend=False,
                err_kws={'alpha':0.75})
    plt.ylabel('population probability')


def diffusion_coef(histones, diff_coef_t_interval):
    times = list(histones.keys())
    msd_dict = TAMSD(histones, times, FONTSIZE, t_limit=diff_coef_time_interval[-1])
    for time in times:
        if len(msd_dict[1][time]) < 1:
            continue

        opts_1 = scipy.optimize.curve_fit(msd_fit_func, msd_dict[0][time], msd_dict[1][time], bounds=(0, [1., 2., 10]))
        est_D, est_alpha, est_noise = opts_1[0][0], opts_1[0][1], opts_1[0][2]
        perr = np.sqrt(np.diag(opts_1[1]))
        diffusion_coefficient = slope_mean(msd_fit_func(msd_dict[0][time], est_D, est_alpha, est_noise), diff_coef_t_interval[0], diff_coef_t_interval[1])
            
        plt.figure()
        plt.title(f'{time}')
        plt.plot(msd_dict[0][time], msd_dict[1][time])
        plt.plot(msd_dict[0][time], msd_fit_func(msd_dict[0][time], est_D, est_alpha, est_noise), c='red')
 
        print(f'---------------------------------- {time} --------------------------------------------------')
        print(f'Function1   ->   Generalized Diff_coef(K):{opts_1[0][0]:>10.7f}')
        print(f'Function1   ->   Std of Generalized Diff_coef(K):{perr[0]:>10.7f}')
        print()
        print(f'Function1   ->   alpha:{opts_1[0][1]:>10.7f}')
        print(f'Function1   ->   Std of alpha:{perr[1]:>10.7f}')
        print()
        print(f'Function1   ->   noise(um^2):{opts_1[0][2]:>10.7f}')
        print(f'Function1   ->   Std of noise(um^2):{perr[2]:>10.7f}')
        print()
        print(f'Calculated diffusion coefficient:{diffusion_coefficient:>10.7f} um^2/s')
        print(f'** Calculated diffusion coefficient for time interval:{diff_coef_t_interval}(s) **')
        print()
        print()


def msd_fit_func(t, D, alpha, noise):
    return 4 * D * t**alpha + noise


def slope_mean(msd_curve, start_index, end_index):
    slopes = []
    start_index *= 100
    end_index *= 100
    end_index = min(end_index, len(msd_curve))

    for delta_t in range(1, int(end_index) - int(start_index)):
        for i in range(int(start_index), int(end_index)-delta_t, 1):
            slope = (msd_curve[i + delta_t] - msd_curve[i]) / (delta_t / 100)
            slopes.append(slope)
    return np.mean(slopes)


if __name__ == '__main__':
    current_path = os.getcwd()
    current_path = current_path.replace('\\', '/')
    condition = 'condition1'
    trxyt_path = f'{current_path}/{condition}/trxyt_files'

    # font size for plots (axis and labels)
    FONTSIZE = 20
    # Register font style
    axis_font = {'family': 'serif', 'size': FONTSIZE}
    plt.rc('font', **axis_font)

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # Element names of plot_list must be same as the folder names
    # Possible names : 'whole_cells', 'before', '15s', '30s', '1min', '2min', '5mins', '10mins', '12mins', '15mins'
    plot_list = ['before', '30s', '1min', '2min']
    color_list = ['blue', 'red', 'blueviolet', 'yellow']

    assert len(plot_list) == len(color_list)
    if len(plot_list) != len(color_list):
        print('Plot list length and Color list length are different, please make them same.')
        exit(1)

    # Minimum trajectory length and bin size settings.
    cutoff = 2  # min trajectory length to analyze
    disp_bin = np.arange(0, 2, 0.01)
    jump_bin = np.arange(0, 0.1, 0.001)  ### you can change jump bin here
    rad_bin = np.arange(0, 2, 0.01)
    diff_coef_time_interval = [0.10, 0.20]
    delta_t = 1
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # Read trajectory files from the path
    histones = read_files_by_each_time(trxyt_path, plotList=plot_list, cutoff=cutoff)
    diffusion_coef(histones, diff_coef_time_interval)

    disp_hist = {key:[] for key in plot_list}
    disp_hist2 = {key:[] for key in plot_list}
    rad_hist = {key:[] for key in plot_list}
    life_time_hist = {key:[] for key in plot_list}

    for t in plot_list:
        displacements = []
        displacements2 = []
        max_radius = [-1] * len(histones[t])
        life_time = np.zeros(5000).astype(int)
        for traj_n, h2b_obj in enumerate(histones[t]):
            trajectory = h2b_obj.get_trajectory()

            for i in range(0, len(trajectory)-delta_t, 1):
                disp = np.sqrt((trajectory[i+delta_t][0] - trajectory[i][0])**2 + (trajectory[i+delta_t][1] - trajectory[i][1])**2)
                displacements.append(disp)
            for i in range(len(trajectory)):
                cur_radius = np.sqrt((trajectory[i][0] - trajectory[0][0])**2 + (trajectory[i][1] - trajectory[0][1])**2)
                max_radius[traj_n] = max(max_radius[traj_n], cur_radius)

            displacements2.append(np.mean(displacements))
            times = (h2b_obj.get_time() * 100).astype(int)
            times = times - int(times[0])
            for life_t in times:
                life_time[life_t] += 1

        disp_hist[t].extend(displacements)
        disp_hist2[t].extend(displacements2)
        rad_hist[t].extend(max_radius)
        life_time_hist[t].extend(life_time[:200])

    # T-test of mean jump distance between times. please check "class_to_show = [0]" to show only the immobile results
    print('\n########   Yuen-Welch Test of mean jump distance  ########')
    trim = 0.2
    for time in range(1, len(plot_list)):
        print(f'result between {plot_list[0]} and {plot_list[time]}')
        print('Result: ',
              stats.ttest_ind(disp_hist2[plot_list[0]], disp_hist2[plot_list[time]], equal_var=False, trim=trim))
        print()

    plt.figure(figsize=(8, 8))
    plt.title('displacement')
    for t, c in zip(plot_list, color_list):
        counts, bins = np.histogram(disp_hist[t], bins=disp_bin, density=True)
        counts /= np.sum(counts)
        plt.hist(bins[:-1], bins, weights=counts, color=c, alpha=0.6, label=f'{t}')
    plt.legend()

    plt.figure(figsize=(8, 8))
    plt.title('life time')
    for t, c in zip(plot_list, color_list):
        plt.plot(np.arange(len(life_time_hist[t]))/100, life_time_hist[t] / np.max(life_time_hist[t]), c=c, label=f'{t}')
    plt.legend()

    plt.figure(figsize=(8, 8))
    plt.title('log life time')
    for t, c in zip(plot_list, color_list):
        plt.plot(np.arange(len(life_time_hist[t]))/100, np.log(life_time_hist[t] / np.max(life_time_hist[t])), c=c, label=f'{t}')
    plt.legend()

    plt.figure('mean jump distance for delta t=1 (1D histogram)', figsize=(8, 8))
    plt.title('mean jump distance for delta t=1 (1D histogram)')
    for t, c in zip(plot_list, color_list):
        counts, bins = np.histogram(disp_hist2[t], bins=jump_bin, density=True)
        counts /= np.sum(counts)
        plt.hist(bins[:-1], bins, weights=counts, color=c, alpha=0.6, label=f'{t}')
    plt.legend()


    new_df = {}
    list1 = []
    list2 = []
    kind = []
    colors = []
    for t, c in zip(plot_list, color_list):
        for element_idx in range(len(disp_hist2[t])):
            list1.append(disp_hist2[t][element_idx])
            list2.append(rad_hist[t][element_idx])
            kind.append(t)
            colors.append(c)

    pd_dict = {'mean_jump_distance':list1, 'max_radial_disp':list2, 'color':colors, 'kind':kind}
    df = pd.DataFrame(data=pd_dict)

    joint1 = sns.jointplot(
        data=df,
        x="mean_jump_distance", y="max_radial_disp", hue='kind', common_norm=False, 
        kind="kde", fill=True, palette=sns.color_palette(palette=color_list, n_colors=len(color_list)), alpha=.7,
        marginal_ticks=True, marginal_kws=dict(fill=True, common_norm=False)
    )
    joint1.fig.suptitle('2D PDF(kde) plot of trajectories')

    kde1 = sns.displot(
        data=df,
        x="mean_jump_distance", hue='kind', common_norm=False, kind="kde",
        fill=True, palette=sns.color_palette(palette=color_list, n_colors=len(color_list)), alpha=.7
    )
    plt.xlim([0, 0.1])  ## set x_lim of 1D PDF plot of trajectories
    kde1.fig.suptitle('1D PDF(kde) plot of trajectories')

    plt.figure()
    hist1 = sns.histplot(
        data=df,
        x="mean_jump_distance", hue='kind', common_norm=False, stat='percent',
        fill=True, palette=sns.color_palette(palette=color_list, n_colors=len(color_list)), alpha=.7
    ).set(title='1D distribution of trjactories')


    """
    for si, plot_list in enumerate([plot_list1]):
        for check_distrib, c_bin in zip([disp_hist, disp_hist2, rad_hist], [disp_bin, jump_bin, rad_bin]):

            for t in plot_list[1:]: 

                print(f'---Between {plot_list[0]} and {t}------')
                kernel1 = stats.gaussian_kde(check_distrib[plot_list[0]])
                kernel2 = stats.gaussian_kde(check_distrib[t])
                givn_data = kernel1.resample([10000]).T.flatten()
                obv_data = kernel2.resample([10000]).T.flatten()

                givn_data = np.histogram(givn_data, bins=c_bin, density=True)
                obv_data = np.histogram(obv_data, bins=c_bin, density=True)

                stats_result = stats.kstest(obv_data[0], givn_data[0], method='exact')
                print(stats_result)
                print('----------------------------------------------')
                plt.figure()
                plt.hist(givn_data[1][:-1], givn_data[1], weights=givn_data[0])
                plt.hist(obv_data[1][:-1], obv_data[1], weights=obv_data[0])
                plt.show()
    """
    plt.show()
