def write_trxyt(file: str, trajectory_list: list, pixel_microns=1.0, frame_rate=1.0):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = ''
            for index, trajectory_obj in enumerate(trajectory_list):
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{index}\t{xpos * pixel_microns:.5f}\t{ypos * pixel_microns:.5f}\t{time * frame_rate:.3f}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def write_trajectory(file: str, trajectory_list: list):
    try:
        with open(file, 'w', encoding="utf-8") as f:
            input_str = 'traj_idx,frame,x,y,z\n'
            for trajectory_obj in trajectory_list:
                for (xpos, ypos, zpos), time in zip(trajectory_obj.get_positions(), trajectory_obj.get_times()):
                    input_str += f'{trajectory_obj.get_index()},{time},{xpos},{ypos},{zpos}\n'
            f.write(input_str)
    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)


def trxyt_to_csv(file: str, pixel_microns=1.0, frame_rate=1.0, to_frame=False) -> dict | list:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    filetypes = ['trxyt', 'trx']
    # Check filetype.
    assert file.strip().split('.')[-1].lower() in filetypes
    new_lines = "traj_idx,frame,x,y,z\n"
    # Read file and store the trajectory and time information in H2B object
    if file.strip().split('.')[-1].lower() in ['trxyt', 'trx']:
        tmp = {}
        with open(file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')
        for line in lines:
            temp = line.split('\t')
            x_pos = float(temp[1].strip()) * pixel_microns
            y_pos = float(temp[2].strip()) * pixel_microns
            z_pos = 0. * pixel_microns
            time_step = float(temp[3].strip()) * frame_rate
            new_lines += f"{temp[0]},{time_step},{x_pos},{y_pos},{z_pos}\n"

    with open(f"{file.split(".trxyt")[0]}_traces.csv", "w") as f:
        f.write(new_lines)
