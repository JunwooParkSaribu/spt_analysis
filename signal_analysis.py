import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from module.fileIO.DataLoad import read_csv, read_h5


    
image_paths = [f'romane_2/1 ROI JF10nM 10ms/1 ROI JF10nM 10ms.tif',
               f'romane_2/2 ROI JF10nM 10ms/2 ROI JF10nM 10ms.tif',
               f'romane_2/5 ROI JF10nM 10ms/5 ROI JF10nM 10ms.tif',
               f'romane_2/3 ROI JF10nM 6ms/3 ROI JF10nM 6ms.tif',
               f'romane_2/6 ROI JF10nM 6ms/6 ROI JF10nM 6ms.tif',
               f'romane_2/7 ROI JF10nM 6ms/7 ROI JF10nM 6ms.tif',

               f'romane_2/22 ROI JF5nM 10ms/22 ROI JF5nM 10ms.tif',
               f'romane_2/23 ROI JF5nM 10ms/23 ROI JF5nM 10ms.tif',
               f'romane_2/19 ROI JF5nM 6ms/19 ROI JF5nM 6ms.tif',
               f'romane_2/20 ROI JF5nM 6ms/20 ROI JF5nM 6ms.tif',
               f'romane_2/21 ROI JF5nM 6ms/21 ROI JF5nM 6ms.tif',

               f'romane_2/13 ROI PAJF50nM 10ms/13 ROI PAJF50nM 10ms.tif',
               f'romane_2/14 ROI PAJF50nM 6ms/14 ROI PAJF50nM 6ms.tif',
               f'romane_2/15 ROI PAJF50nM 6ms/15 ROI PAJF50nM 6ms.tif',]

trace_paths = [f'romane_2/1 ROI JF10nM 10ms/1 ROI JF10nM 10ms_traces_biadd.h5',
               f'romane_2/2 ROI JF10nM 10ms/2 ROI JF10nM 10ms_traces_biadd.h5',
               f'romane_2/5 ROI JF10nM 10ms/5 ROI JF10nM 10ms_traces_biadd.h5',
               f'romane_2/3 ROI JF10nM 6ms/3 ROI JF10nM 6ms_traces_biadd.h5',
               f'romane_2/6 ROI JF10nM 6ms/6 ROI JF10nM 6ms_traces_biadd.h5',
               f'romane_2/7 ROI JF10nM 6ms/7 ROI JF10nM 6ms_traces_biadd.h5',

               f'romane_2/22 ROI JF5nM 10ms/22 ROI JF5nM 10ms_traces_biadd.h5',
               f'romane_2/23 ROI JF5nM 10ms/23 ROI JF5nM 10ms_traces_biadd.h5',
               f'romane_2/19 ROI JF5nM 6ms/19 ROI JF5nM 6ms_traces_biadd.h5',
               f'romane_2/20 ROI JF5nM 6ms/20 ROI JF5nM 6ms_traces_biadd.h5',
               f'romane_2/21 ROI JF5nM 6ms/21 ROI JF5nM 6ms_traces_biadd.h5',

               f'romane_2/13 ROI PAJF50nM 10ms/13 ROI PAJF50nM 10ms_traces_biadd.h5',
               f'romane_2/14 ROI PAJF50nM 6ms/14 ROI PAJF50nM 6ms_traces_biadd.h5',
               f'romane_2/15 ROI PAJF50nM 6ms/15 ROI PAJF50nM 6ms_traces_biadd.h5',]

colors = [f'red', f'tomato', f'chocolate', f'darkorange',
          f'gold', f'yellow', f'lawngreen', f'lightgreen', 
          f'green', f'lightseagreen', f'aqua', f'deepskyblue',
          f'dodgerblue', f'royalblue', f'navy', f'slateblue',
          f'darkviolet', f'magenta', f'deeppink', f'lightpink'
          f'black']


sig_lines = 7
states = [0, 1]
ncols = len(states)


def read_tif(filepath):
    with tifffile.TiffFile(filepath) as tif:
        imgs = tif.asarray()
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
    mins = np.min(imgs, axis=(1, 2)).reshape(-1, 1, 1)
    maxs = np.max(imgs, axis=(1, 2)).reshape(-1, 1, 1)
    imgs = (imgs - mins) / (maxs - mins)
    return imgs


fig, axs = plt.subplots(nrows=sig_lines, ncols=ncols, figsize=(18, 9))
plt.setp(axs, ylim=[0.0, 1.0])
backgrounds = []

if not os.path.exists('./tmp'):
    os.mkdir('./tmp')

for ax_col, (img_path, trace_path, color) in enumerate(zip(image_paths, trace_paths, colors)):
    tmp_bg_path= f"./tmp/{img_path.split('/')[-1].split('.tif')[0]}_bg_intensities.npz"
    images = read_tif(img_path)
    df, meta = read_h5(trace_path)
    mask = np.ones_like(images, dtype=bool)

    for st in states:
        filtered_df = df[df['state']==st]
        xs = filtered_df['x'].to_numpy()
        ys = filtered_df['y'].to_numpy()
        frames = filtered_df['frame'].to_numpy() - 1
        signal_rows = [[] for _ in range(sig_lines)]

        for frame, x, y in zip(frames, xs, ys):
            col_center, row_center = int(np.round(x)), int(np.round(y))
            col_range = [col_center - sig_lines//2, col_center + sig_lines//2 + 1]
            row_range = [row_center - sig_lines//2, row_center + sig_lines//2 + 1]
            target_window = images[frame, row_range[0]: row_range[1], col_range[0]: col_range[1]]
            if not os.path.exists(tmp_bg_path):
                mask[frame, max(0, row_range[0]): min(images.shape[1], row_range[1]), max(0, col_range[0]): min(images.shape[1], col_range[1])] = False
            if target_window.shape[0] == sig_lines and target_window.shape[1] == sig_lines:
                target_window = (target_window - np.min(target_window)) / (np.max(target_window) - np.min(target_window))
                for idx in range(sig_lines):
                    signal_rows[idx].append(target_window[idx])

        signal_rows = np.array(signal_rows)
        sig_row_means = np.mean(signal_rows, axis=(1))
        sig_row_stds = np.std(signal_rows, axis=(1))
        for r_idx in range(len(signal_rows)):
            #for line in signal_rows[r_idx]:
            #    axs[r_idx][0].plot(line, alpha=0.01, c='red')
            axs[r_idx][st].plot(sig_row_means[r_idx], c=color, alpha=0.55, label=img_path)
            eb = axs[r_idx][st].errorbar(np.arange(sig_lines), sig_row_means[r_idx], yerr=sig_row_stds[r_idx], capsize=10, alpha=0.4, c=color)
            axs[r_idx][st].set_xticks([])
          
    if not os.path.exists(tmp_bg_path):
        bg_signals = images[mask, ...]
        np.savez(tmp_bg_path, data=bg_signals)
    else:
        with np.load(tmp_bg_path) as bg:
            bg_signals = bg['data']

    backgrounds.append([np.mean(bg_signals), np.std(bg_signals)])

backgrounds = np.array(backgrounds)
plt.figure()
plt.plot(np.arange(len(backgrounds)), backgrounds[:,0], c='red', alpha=1.0)
plt.errorbar(np.arange(len(backgrounds)), backgrounds[:, 0], yerr=backgrounds[:, 1], capsize=7, alpha=1.0, c='red')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
