import pandas as pd
import numpy as np
import cv2


def trajectory_visualization(data:pd.DataFrame, cutoff:int):
    resolution_factor = 3
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
            pts = np.array([[int(x * 10**resolution_factor), int(y * 10**resolution_factor)] for x, y in zip(single_traj['x'], single_traj['y'])], np.int32)
            for i in range(len(pts)-1):
                prev_pt = pts[i]
                next_pt = pts[i+1]
                cv2.line(image, prev_pt, next_pt, color, thickness)
    cv2.imwrite('./visualization.png', image)
