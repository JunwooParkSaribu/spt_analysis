import numpy as np


def trajectory_rotation(histones, nb: int) -> dict:
    if 360 % nb != 0:
        print('360%nb should be 0')
        raise Exception

    if type(histones) == list:
        h = {}
        for histone_chunk in histones:
            h |= histone_chunk
        histones = h

    thetas = [theta * (int(360 / nb)) for theta in range(nb)]
    rotation = lambda theta: np.array([
        [np.cos((np.pi * theta) / 180), np.sin((np.pi * theta) / 180)],
        [-np.sin((np.pi * theta) / 180), np.cos((np.pi * theta) / 180)]
    ])
    rotation_matrix = [rotation(theta) for theta in thetas]

    augmented_histones = {}
    for histone in histones:
        for theta, rot_mat in zip(thetas, rotation_matrix):
            trajectory = histones[histone].get_trajectory().copy()
            initial_positions = trajectory[0].copy()
            rotated_histone = histones[histone].copy()
            rotated_histone.set_id(f'{rotated_histone.get_id()}_{theta}deg')
            trajectory_temp = []

            for i in range(len(trajectory)):
                trajectory[i] -= initial_positions
            for traj in trajectory:
                trajectory_temp.append(np.dot(rot_mat, traj))
            for i in range(len(trajectory_temp)):
                trajectory_temp[i] += initial_positions

            rotated_histone.set_trajectory(trajectory_temp)
            augmented_histones[f'{rotated_histone.get_file_name()}@{rotated_histone.get_id()}'] = rotated_histone
    del histones
    return augmented_histones


def distance(histones):
    distances = {}
    for histone in histones:
        distances[histone] = []

    for histone in histones:
        dist = 0
        for i in range(len(histones[histone]) - 1):
            x_distance = histones[histone][i+1][0] - histones[histone][i][0]
            y_distance = histones[histone][i+1][1] - histones[histone][i][1]
            dist += np.sqrt(x_distance**2 + y_distance**2)
        t = histones[histone][-1][2] - histones[histone][0][2]
        distances[histone].append(dist)
        distances[histone].append(t)
    return distances


def seg_distance(histones):
    distances = {}
    for histone in histones:
        distances[histone] = []

    for histone in histones:
        for i in range(len(histones[histone]) - 1):
            x_distance = histones[histone][i+1][0] - histones[histone][i][0]
            y_distance = histones[histone][i+1][1] - histones[histone][i][1]
            dist = np.sqrt(x_distance**2 + y_distance**2)
            distances[histone].append(dist)
    return distances


def displacement(histones):
    displacements = {}
    for histone in histones:
        displacements[histone] = []

    for histone in histones:
        trajectory = histones[histone].get_trajectory()
        for i in range(1, len(trajectory)):
            x_displacement = trajectory[i-1][0] - trajectory[i][0]
            y_displacement = trajectory[i-1][1] - trajectory[i][1]
            displacements[histone].append(np.sqrt(x_displacement ** 2 + y_displacement ** 2))
    return displacements


def velocity(histones):
    histone_velocity = {}
    for histone in histones:
        histone_velocity[histone] = []

    for histone in histones:
        for trajec_num in range(histones[histone].get_len_trajectory() - 1):
            trajectory = histones[histone].get_trajectory()
            time = histones[histone].get_time()
            x_distance = trajectory[trajec_num + 1][0] - trajectory[trajec_num][0]
            y_distance = trajectory[trajec_num + 1][1] - trajectory[trajec_num][1]
            t = time[trajec_num + 1] - time[trajec_num]
            if t == 0:
                print('trajectory speed calculation warning:', histone)
            histone_velocity[histone].append(np.sqrt(x_distance**2 + y_distance**2)/t)
    return histone_velocity


def accumulate(histone):
    acc_histone = []
    acc = 0
    for velocity in histone:
        acc += velocity[0]
        acc_histone.append([acc])
    return acc_histone


def check_balls(histones, radius=0.45, density=0.5) -> dict:
    histones_balls = {}
    for histone in histones:
        n_balls = 0
        hybrid_flag = 0
        all_trajec = histones[histone].get_trajectory()
        all_trajec_n = len(histones[histone].get_trajectory())
        for i in range(len(all_trajec)):
            trajec_density = 0
            pos = all_trajec[i]
            for j in range(len(all_trajec)):
                next_pos = all_trajec[j]
                if np.sqrt((next_pos[0] - pos[0])**2 + (next_pos[1] - pos[1])**2) < radius:
                    trajec_density += 1
                else:
                    hybrid_flag = 1
            if trajec_density == all_trajec_n:
                if histones[histone].get_time_duration() < 20 and histones[histone].get_max_radius() > 0.2:
                    n_balls = 0  # mobile
                    break
                else:
                    n_balls = 1  # immobile
                    break
            if trajec_density/all_trajec_n > density and all_trajec_n > 15:
                n_balls += 1
        histones_balls[histone] = [n_balls, hybrid_flag]
        del all_trajec
    return histones_balls


def calcul_max_radius(histones):
    for histone in histones:
        trajectories = histones[histone].get_trajectory()
        first_position = trajectories[0]
        max_r = 0
        for trajectory in trajectories:
            dist = np.sqrt((first_position[0] - trajectory[0])**2 + (first_position[1] - trajectory[1])**2)
            max_r = max(max_r, dist)
        histones[histone].set_max_radius(max_r)


def diff_coef(histones):
    for histone in histones:
        trajectories = histones[histone].get_trajectory()
        times = histones[histone].get_time()
        coef = []
        for i in range(1, len(trajectories)):
            x_disp = trajectories[i][0] - trajectories[i-1][0]
            y_disp = trajectories[i][1] - trajectories[i-1][1]
            disp = np.sqrt(x_disp**2 + y_disp**2)
            coef.append(disp**2 / (4 * (times[i]-times[i-1])))
        histones[histone].set_diff_coef(coef)
