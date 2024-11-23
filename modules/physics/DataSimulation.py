import numpy as np
from histone.H2B import H2B


def make_immobile(histones, nb=5, radius=0.4, max_distance=0.085, cond=(10, 150)):
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(cond[0], cond[1]))
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < n_trajectory:
            x = np.random.uniform(prev_xy[0]-max_distance, prev_xy[0]+max_distance)
            y = np.random.uniform(prev_xy[1]-max_distance, prev_xy[1]+max_distance)
            xy = np.array([x, y])
            if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                if len(trajectory) > 1:
                    direction = xy - prev_xy
                    if np.dot(direction, prev_direction) < 0:
                        trajectory.append(xy)
                        prev_direction = direction
                        prev_xy = xy
                else:
                    trajectory.append(xy)
                    prev_direction = xy - prev_xy
                    prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(n_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(0)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b


def make_mobile(histones, nb=5, max_distance=0.45, cond=(3, 15)):
    dist_for_immobile = 0.12
    for i in range(nb):
        h2b = H2B()
        n_trajectory = int(np.random.uniform(cond[0], cond[1]))
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < n_trajectory:
            x = np.random.uniform(prev_xy[0]-max_distance, prev_xy[0]+max_distance)
            y = np.random.uniform(prev_xy[1]-max_distance, prev_xy[1]+max_distance)
            xy = np.array([x, y])
            if np.sqrt((xy - prev_xy)[0] ** 2 + (xy - prev_xy)[1]) < dist_for_immobile:
                continue
            trajectory.append(xy)
            prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(n_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(2)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b


def make_hybrid(histones, nb=5, radius=0.4, max_dist_immobile=0.085, max_dist_mobile=0.45, type=0):
    for i in range(nb):
        h2b = H2B()
        ball_trajectory = int(np.random.uniform(20, 150))
        intermediate_trajectory = int(np.random.randint(3, 7))
        total_trajectory = ball_trajectory + intermediate_trajectory
        trajectory = [[10, 10]]
        prev_xy = trajectory[0]

        while len(trajectory) < total_trajectory:
            if type == 0:
                if len(trajectory) < ball_trajectory/2:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy
                        else:
                            trajectory.append(xy)
                            prev_direction = xy - prev_xy
                            prev_xy = xy
                elif len(trajectory) < (ball_trajectory/2 + intermediate_trajectory):
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    flag = 1
                    for traj in trajectory:
                        if np.sqrt((traj[0] - x) ** 2 + (traj[1] - y) ** 2) < radius:
                            flag = 0
                    if flag == 1:
                        xy = np.array([x, y])
                        trajectory.append(xy)
                        prev_direction = xy - prev_xy
                        prev_xy = xy
                        new_center = prev_xy.copy()
                else:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - new_center[0])**2 + (xy[1] - new_center[1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy

            elif type == 1:
                if len(trajectory) < ball_trajectory:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - trajectory[0][0])**2 + (xy[1] - trajectory[0][1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy
                        else:
                            trajectory.append(xy)
                            prev_direction = xy - prev_xy
                            prev_xy = xy
                else:
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    flag = 1
                    if np.sqrt((trajectory[-1][0] - x) ** 2 + (trajectory[-1][1] - y) ** 2) < radius:
                        flag = 0
                    if flag == 1:
                        xy = np.array([x, y])
                        trajectory.append(xy)
                        prev_direction = xy - prev_xy
                        prev_xy = xy

            else:
                if len(trajectory) < intermediate_trajectory+1:
                    x = np.random.uniform(prev_xy[0] - max_dist_mobile, prev_xy[0] + max_dist_mobile)
                    y = np.random.uniform(prev_xy[1] - max_dist_mobile, prev_xy[1] + max_dist_mobile)
                    flag = 1
                    for traj in trajectory:
                        if np.sqrt((traj[0] - x) ** 2 + (traj[1] - y) ** 2) < radius:
                            flag = 0
                    if flag == 1:
                        xy = np.array([x, y])
                        trajectory.append(xy)
                        prev_direction = xy - prev_xy
                        prev_xy = xy
                        new_center = prev_xy.copy()
                else:
                    x = np.random.uniform(prev_xy[0]-max_dist_immobile, prev_xy[0]+max_dist_immobile)
                    y = np.random.uniform(prev_xy[1]-max_dist_immobile, prev_xy[1]+max_dist_immobile)
                    xy = np.array([x, y])
                    if np.sqrt((xy[0] - new_center[0])**2 + (xy[1] - new_center[1])**2) < radius:
                        if len(trajectory) > 1:
                            direction = xy - prev_xy
                            if np.dot(direction, prev_direction) < 0:
                                trajectory.append(xy)
                                prev_direction = direction
                                prev_xy = xy

        trajectory = np.array(trajectory)
        times = (np.arange(total_trajectory)+1)*0.01
        h2b.set_trajectory(trajectory)
        h2b.set_time(times)
        h2b.set_file_name('simulated_data')
        h2b.set_manuel_label(1)

        if len(histones) != 0:
            id = str(np.sort(np.array(list(histones.keys())).astype(int))[-1] + 1)
            h2b.set_id(id)
            histones[id] = h2b
        else:
            h2b.set_id(str(0))
            histones[str(0)] = h2b


def make_simulation_data(number=4200):
    histones = {}

    # make immobile H2Bs
    make_immobile(histones, nb=int(number/6), radius=0.02, max_distance=0.12)
    make_immobile(histones, nb=int(number/6), radius=0.05, max_distance=0.12)
    make_immobile(histones, nb=int(number/6), radius=0.1, max_distance=0.12)
    make_immobile(histones, nb=int(number/6), radius=0.2, max_distance=0.12)
    make_immobile(histones, nb=int(number/6), radius=0.3, max_distance=0.12)
    make_immobile(histones, nb=int(number/6), radius=0.4, max_distance=0.12)
    print("Immobile generated")

    # make hybrid H2Bs
    make_hybrid(histones, nb=int(number/3), radius=0.2, max_dist_immobile=0.12,
                max_dist_mobile=0.45, type=0)
    make_hybrid(histones, nb=int(number/3), radius=0.2, max_dist_immobile=0.12,
                max_dist_mobile=0.45, type=1)
    make_hybrid(histones, nb=int(number/3), radius=0.2, max_dist_immobile=0.12,
                max_dist_mobile=0.45, type=2)
    print("Hybrid generated")

    # make mobile H2Bs
    make_mobile(histones, nb=int(number), max_distance=0.45)
    print("Mobile generated")
    return histones
