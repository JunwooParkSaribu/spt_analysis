import networkx as nx
import itertools as it
import pandas as pd
import numpy as np
import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
from tqdm import tqdm
from roifile import roifile


def trajectory_visualization_old(output_path:str, data:pd.DataFrame, cutoff:int):
    resolution_factor = 2
    thickness = 2

    min_x = data['x'].min()
    min_y = data['y'].min()
    max_x = data['x'].max()
    max_y = data['y'].max()
    x_width = int(((max_x - min_x) * 10**resolution_factor))
    y_width = int(((max_y - min_y) * 10**resolution_factor))
    image = np.ones((y_width, x_width, 3)).astype(np.uint8)
    for traj_idx in data['traj_idx'].unique():
        single_traj = data[data['traj_idx'] == traj_idx]
        if len(single_traj) >= cutoff:
            if single_traj['state'].iloc[0] == 0:
                color = (0, 0, 255) #in BGR
            elif single_traj['state'].iloc[0] == 1:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            pts = np.array([[int((x - min_x) * 10**resolution_factor), int((y - min_y) * 10**resolution_factor)] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
            for i in range(len(pts)-1):
                prev_pt = pts[i]
                next_pt = pts[i+1]
                cv2.line(image, prev_pt, next_pt, color, thickness)
    cv2.imwrite(f'{output_path}/visualization.png', image)


def trajectory_visualization(original_data:pd.DataFrame, analysis_data1:pd.DataFrame, cutoff:int, pixelmicron:float, resolution_multiplier=20, roi='') -> np.ndarray:
    print("** visualizing trajectories... **")
    scale = resolution_multiplier
    thickness = 1
    color_maps = {}
    color_maps_plot = {}

    min_x = original_data['x'].min()
    min_y = original_data['y'].min()
    max_x = original_data['x'].max()
    max_y = original_data['y'].max()
    x_width = int(((max_x - min_x) * scale))
    y_width = int(((max_y - min_y) * scale))
    image = np.ones((y_width, x_width, 3)).astype(np.uint8)

    for traj_idx in tqdm(original_data['traj_idx'].unique(), ncols=120, desc=f'Visualization', unit='trajectory'):
        single_traj = original_data[original_data['traj_idx'] == traj_idx]
        corresponding_ids = analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]
        if np.sum(corresponding_ids) > 0:
            traj_color = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['color'].iloc[0]
            state = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['state'].iloc[0]
            if traj_color not in color_maps and traj_color != 'yellow':
                color_maps[traj_color] = str(state)
                color_maps_plot[state] = traj_color
            if len(single_traj) >= cutoff:
                traj_color = mcolors.to_rgb(traj_color)
                traj_color = (int(traj_color[2]*255), int(traj_color[1]*255), int(traj_color[0]*255))  # BGR color for cv2
                pts = np.array([[int((x - min_x) * scale), int((y - min_y) * scale)] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
                for i in range(len(pts)-1):
                    prev_pt = pts[i]
                    next_pt = pts[i+1]
                    cv2.line(image, prev_pt, next_pt, traj_color, thickness)

    cv2.line(image, [int(max(0, x_width - 2*scale - int(scale/pixelmicron))), int(max(0, y_width - 2*scale))], [int(max(0, x_width - 2*scale)) , int(max(0, y_width - 2*scale))], (255, 255, 255), 6)
    if len(roi) > 4:
        from roifile import ImagejRoi
        contours = ImagejRoi.fromfile(roi).coordinates().astype(np.int32)
        for i in range(len(contours) - 1):
            prev_pt = (np.array([contours[i][0] - min_x, contours[i][1] - min_y])*scale).astype(int)
            next_pt = (np.array([contours[i+1][0] - min_x, contours[i+1][1] - min_y])*scale).astype(int)
            cv2.circle(image, next_pt, thickness+1, (128, 128, 128), -1)
            
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_maps['yellow'] = 'transitioning'
    color_maps_plot['transitioning'] = 'yellow'
    patches =[mpatches.Patch(color=c,label=color_maps[c]) for c in color_maps]
    return image, patches, color_maps, color_maps_plot


def quantification(window_size):
    x = np.arange(-(window_size-1)/2, (window_size+1)/2)
    y = np.arange(-(window_size-1)/2, (window_size+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size * window_size, 2)
    return grid.astype(np.float32)


def trajectory_visualization_tmp(original_data:pd.DataFrame, analysis_data1:pd.DataFrame, cutoff:int, pixelmicron:float, resolution_multiplier=20) -> np.ndarray:
    print("** visualizing trajectories... **")
    import matplotlib.pyplot as plt
    from PIL import Image
    scale = resolution_multiplier
    thickness = 1
    color_maps = {}
    color_maps_plot = {}
    winsize = 55
    cov_std = 55
    margin_pixel = 200

    min_x = original_data['x'].min()
    min_y = original_data['y'].min()
    max_x = original_data['x'].max()
    max_y = original_data['y'].max()
    x_width = int(((max_x - min_x) * scale) + margin_pixel)
    y_width = int(((max_y - min_y) * scale) + margin_pixel)
    image = np.ones((y_width, x_width)).astype(np.float64)


    mycmap = plt.get_cmap('magma', lut=None)
    color_seq = [mycmap(i)[:3] for i in range(mycmap.N)]
    all_coords = np.vstack([original_data['x'].to_numpy(), original_data['y'].to_numpy()]).T
    all_coords[:,0] = all_coords[:,0] - min_x
    all_coords[:,1] = all_coords[:,1] - min_y
    all_coords = np.round(all_coords * scale)

    template = np.ones((1, (winsize)**2, 2), dtype=np.float32) * quantification(winsize)
    template = (np.exp((-1./2) * np.sum(template @ np.linalg.inv([[cov_std, 0], [0, cov_std]]) * template, axis=2))).reshape([winsize, winsize])

    for roundup_coord in all_coords:
        coord_col = int(roundup_coord[0] + margin_pixel//2)
        coord_row = int(roundup_coord[1] + margin_pixel//2)
        row = min(max(0, coord_row), image.shape[0])
        col = min(max(0, coord_col), image.shape[1])
        try:
            image[row - winsize//2: row + winsize//2 + 1, col - winsize//2: col + winsize//2 + 1] += template
        except:
            pass
    
    img_min, img_max = np.quantile(image, [0.01, 1.0])
    image = np.minimum(image, np.ones_like(image) * img_max)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)
    image = plt.imshow(image, cmap=mycmap, origin='upper')
    image = image.make_image(renderer=None, unsampled=True)[0][:,:,:3]
    image = np.array(Image.fromarray(image))
    image = np.array(image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    print(image.shape)
    #image.save(f'{output_dir}_loc_{dim}d_density.png', dpi=(300, 300))

    
    n_lines = 100
    cmap = mpl.colormaps['jet']
    colors = cmap(np.linspace(0, 1, n_lines))[::-1][0:75]

    for traj_idx in tqdm(original_data['traj_idx'].unique(), ncols=120, desc=f'Visualization', unit='trajectory'):
        single_traj = original_data[original_data['traj_idx'] == traj_idx]
        corresponding_ids = analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]
        tmpx = single_traj['x'].to_numpy() 
        tmpy = single_traj['y'].to_numpy()
        dips = np.mean(np.sqrt((tmpx[1:] - tmpx[:-1])**2 + (tmpy[1:] - tmpy[:-1])**2))
        if np.sum(corresponding_ids) > 0:
            traj_color = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['color'].iloc[0]
            state = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['state'].iloc[0]
            if traj_color not in color_maps and traj_color != 'yellow':
                color_maps[traj_color] = str(state)
                color_maps_plot[state] = traj_color
            if len(single_traj) >= cutoff:
                traj_color = mcolors.to_rgb(traj_color)
                traj_color = (int(traj_color[2]*255), int(traj_color[1]*255), int(traj_color[0]*255))  # BGR color for cv2
                pts = np.array([[int((x - min_x) * scale) + margin_pixel//2, int((y - min_y) * scale) + margin_pixel//2] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
                for i in range(len(pts)-1):
                    prev_pt = pts[i]
                    next_pt = pts[i+1]
                    disp_ = np.sum((pts[i+1]/scale - pts[i]/scale)**2)
                    if disp_ < 0.4:
                        mymax = 0.70
                        color_rgba = colors[int(min(1, (dips / mymax)) * (len(colors)-1))]
                        target_color = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
                        cv2.line(image, prev_pt, next_pt, target_color, thickness)
    cv2.line(image, [int(max(0, x_width - 5*scale - int(scale/pixelmicron))), int(max(0, y_width - 5*scale))], [int(max(0, x_width - 5*scale)) , int(max(0, y_width - 5*scale))], (255, 255, 255), 6)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_maps['yellow'] = 'transitioning'
    color_maps_plot['transitioning'] = 'yellow'
    patches =[mpatches.Patch(color=c,label=color_maps[c]) for c in color_maps]
    return image.copy(), patches, color_maps, color_maps_plot


def draw_labeled_multigraph(G, attr_names, cmap=None, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """

    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.20] * 10)]
    G.add_node('dummy')  # dummy node for circular layout
    graph_node_list = [node for node in G.nodes if node != 'dummy']
    
    if cmap is not None:
        d_swap = {v: k for k, v in cmap.items()}
        graph_node_colors = [d_swap[str(node)] for node in graph_node_list]
    else:
        graph_node_colors = ['blue'] * len(graph_node_list)

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=graph_node_list, node_color=graph_node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax, arrowsize=50, min_source_margin=30, min_target_margin=30, arrowstyle='-|>'
    )

    labels = {}
    for *edge, attrs in G.edges(data=True):
        for attr_name in attr_names:
            if tuple(edge) not in labels:
                labels[tuple(edge)] = f'{attr_name}={attrs[attr_name]}'
            else:
                labels[tuple(edge)] += f'\n{attr_name}={attrs[attr_name]}'

    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0.9},
        rotate=False,
        ax=ax,
    )

    if 'count' in attr_names:
        total_ = 0
        for edze in G.edges:
            total_ += G[edze[0]][edze[1]]['count']
        ax.set_title(f'Total number of events: {total_}')
