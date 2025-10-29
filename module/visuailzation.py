import networkx as nx
import itertools as it
import pandas as pd
import numpy as np
import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
from tqdm import tqdm


def trajectory_visualization(original_data:pd.DataFrame, analysis_data1:pd.DataFrame, 
                             cutoff:list, pixelmicron:float, resolution_multiplier=80, 
                             roi='', scalebar=True, arrow=False, color_for_roi=False, thickness = 3) -> np.ndarray:
    print("** visualizing trajectories... **")
    scale = resolution_multiplier
    color_maps = {}
    color_maps_plot = {}
    roi_center = None

    min_x = original_data['x'].min()
    min_y = original_data['y'].min()
    max_x = original_data['x'].max()
    max_y = original_data['y'].max()
    x_width = int(((max_x - min_x) * scale))
    y_width = int(((max_y - min_y) * scale))
    image = np.ones((y_width, x_width, 4)).astype(np.uint8)

    if len(roi) > 4:
        from roifile import ImagejRoi
        contours = ImagejRoi.fromfile(roi).coordinates().astype(np.int32)
        roi_center = (np.array([contours[0][0] - min_x, contours[0][1] - min_y])*scale).astype(int)
        for i in range(len(contours) - 1):
            prev_pt = (np.array([contours[i][0] - min_x, contours[i][1] - min_y])*scale).astype(int)
            next_pt = (np.array([contours[i+1][0] - min_x, contours[i+1][1] - min_y])*scale).astype(int)
            cv2.circle(image, next_pt, thickness+1, (128, 128, 128, 255), -1)
            roi_center += next_pt
        roi_center = roi_center.astype(float)
        roi_center /= len(contours)

    for traj_idx in tqdm(original_data['traj_idx'].unique(), ncols=120, desc=f'Visualization', unit='trajectory'):
        single_traj = original_data[original_data['traj_idx'] == traj_idx]
        corresponding_ids = analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]
        if np.sum(corresponding_ids) > 0:
            traj_color = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['color'].iloc[0]
            state = analysis_data1[analysis_data1['traj_id'] == single_traj['traj_idx'].iloc[0]]['state'].iloc[0]
            if traj_color not in color_maps and traj_color != 'yellow':
                color_maps[traj_color] = str(state)
                color_maps_plot[state] = traj_color
            if cutoff[0] <= len(single_traj) <= cutoff[1]:
                traj_color = mcolors.to_rgb(traj_color)
                traj_color = (int(traj_color[2]*255), int(traj_color[1]*255), int(traj_color[0]*255), 255)  # BGR color for cv2
                pts = np.array([[int((x - min_x) * scale), int((y - min_y) * scale)] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
                for i in range(len(pts)-1):
                    prev_pt = pts[i]
                    next_pt = pts[i+1]
                    if arrow:
                        if color_for_roi and roi_center is not None:
                            if np.sqrt(np.sum((next_pt - roi_center)**2)) < np.sqrt(np.sum((prev_pt - roi_center)**2)):
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 0, 255, 255), thickness)
                            else:
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 255, 255, 255), thickness)
                            """
                            from module.preprocessing import dot_product_angle
                            angle = dot_product_angle(next_pt - prev_pt, roi_center - prev_pt)
                            if 0 < angle < 90 or 270 < angle < 360:
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 0, 255), thickness)
                            else:
                                cv2.arrowedLine(image, prev_pt, next_pt, (0, 255, 255), thickness)
                            """
                        else:
                            cv2.arrowedLine(image, prev_pt, next_pt, traj_color, thickness)
                    else:
                        cv2.line(image, prev_pt, next_pt, traj_color, thickness)
    if scalebar:
        cv2.line(image, [int(max(0, x_width - 2*scale - int(scale/pixelmicron))), int(max(0, y_width - 2*scale))], [int(max(0, x_width - 2*scale)) , int(max(0, y_width - 2*scale))], (255, 255, 255, 255), 6)
            
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_maps['yellow'] = 'transitioning'
    color_maps_plot['transitioning'] = 'yellow'
    patches = [mpatches.Patch(color=c,label=color_maps[c]) for c in color_maps]
    return image, patches, color_maps, color_maps_plot


def quantification(window_size):
    x = np.arange(-(window_size-1)/2, (window_size+1)/2)
    y = np.arange(-(window_size-1)/2, (window_size+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size * window_size, 2)
    return grid.astype(np.float32)


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
