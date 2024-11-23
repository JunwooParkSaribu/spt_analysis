import sys
import numpy as np
import cv2
import tifffile


def trxyt_to_trajs(file: str, cutoff: 0, filetype='trxyt') -> dict:
    """
    @params : filename(String), cutoff value(Integer)
    @return : dictionary of H2B objects(Dict)
    Read a single trajectory file and return the dictionary.
    key is consist of filename@id and the value is H2B object.
    Dict only keeps the histones which have the trajectory length longer than cutoff value.
    """
    # Check filetype.
    assert file.strip().split('.')[-1] == filetype
    histone_trajs = {}
    trajectory = {}
    time = {}
    # Read file and store the trajectory and time information in H2B object
    try:

        max_time = -99999
        min_x = 99999
        min_y = 99999
        max_x = -99999
        max_y = -99999

        with open(file, 'r', encoding="utf-8") as f:
            input = f.read()
        lines = input.strip().split('\n')
        file_name = file.strip().split('\\')[-1].split('/')[-1].strip()
        for line in lines:
            temp = line.split('\t')
            id = temp[0].strip()
            x_pos = float(temp[1].strip())
            y_pos = float(temp[2].strip())
            time_step = int(float(temp[3].strip()) * 100)
            max_time = max(max_time, time_step)
            min_x = min(min_x, x_pos)
            min_y = min(min_y, y_pos)
            max_x = max(max_x, x_pos)
            max_y = max(max_y, y_pos)

            if id in trajectory:
                trajectory[id].append([x_pos, y_pos])
                time[id].append(time_step)
            else:
                trajectory[id] = [[x_pos, y_pos]]
                time[id] = [time_step]

        for histone in trajectory:
            if len(trajectory[histone]) >= cutoff:
                histone_trajs[histone] = {}
                histone_trajs[histone]['positions'] = {}
                histone_trajs[histone]['times'] = {}
                
                histone_trajs[histone]['positions'] = np.array(trajectory[histone])
                histone_trajs[histone]['times'] = np.array(time[histone])
                histone_trajs[histone]['colors'] = (np.random.randint(0,256)/256.,
                                                    np.random.randint(0,256)/256.,
                                                    np.random.randint(0,256)/256.)

    except Exception as e:
        print(f"Unexpected error, check the file: {file}")
        print(e)

    finally:
        del trajectory
        del time
        return histone_trajs, min_x, max_x, min_y, max_y, max_time


def make_image_seqs(trxyt_file, output_dir, cutoff=0, add_index=False, resolution=20):
    font_scale = 0.1 * 2
    amp = resolution
    if amp >= 50:
        thickness = 2
    else:
        thickness = 1

    trajs, min_x, max_x, min_y, max_y, max_time = trxyt_to_trajs(trxyt_file, cutoff=cutoff)
    img_stacks = np.zeros([max_time, int((max_y - min_y)*amp), int((max_x - min_x)*amp), 3], dtype=np.uint8)
    for h2b_id in trajs:
        trajs[h2b_id]['positions'][:,0] -= min_x
        trajs[h2b_id]['positions'][:,1] -= min_y

    result_stack = []
    for img, frame in zip(img_stacks, np.arange(max_time)+1):
        print(f'frame {frame} processing')
        img = np.ascontiguousarray(img)
        img_org = img.copy()
        overlay = np.zeros(img.shape)

        for h2b_id in trajs:
            times = trajs[h2b_id]['times']
            if times[-1] < frame:
                continue
            indices = [i for i, time in enumerate(times) if time <= frame]

            xy = np.array([[int(np.around(x * amp)), int(np.around((max_y - y) * amp))]
                            for x, y in trajs[h2b_id]['positions'][indices]], np.int32)
         
            img_poly = cv2.polylines(overlay, [xy],
                                    isClosed=False,
                                    color=(trajs[h2b_id]['colors'][0],
                                           trajs[h2b_id]['colors'][1],
                                           trajs[h2b_id]['colors'][2]),
                                           thickness=thickness)

            if len(indices) > 0:
                if add_index:
                    cv2.putText(overlay, f'[{times[indices[0]]},{times[indices[-1]]}]',
                                org=[xy[0][0], xy[0][1] + 8], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale,
                                color=(trajs[h2b_id]['colors'][0],
                                       trajs[h2b_id]['colors'][1],
                                       trajs[h2b_id]['colors'][2]))
                    cv2.putText(overlay, f'{h2b_id}', org=xy[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale,
                                color=(trajs[h2b_id]['colors'][0],
                                       trajs[h2b_id]['colors'][1],
                                       trajs[h2b_id]['colors'][2]))
        overlay = img_org + overlay
        overlay = np.minimum(np.ones_like(overlay), overlay)
        hstacked_img = overlay
        result_stack.append(hstacked_img)
    print('Saving video...')
    result_stack = (np.array(result_stack)*255).astype(np.uint8)
    tifffile.imwrite(output_dir, data=result_stack, imagej=True)
    print('Finished')


if __name__ == '__main__':
    ###############################################################################

    # higher number creates higher resolution. (10 = lowest, 100 = highest)
    # you can increase as much as you can unless your RAM allows
    # For example, 40 resolution for 5000 frames generate a 9GB video.
    resolution = 20

    # cutoff of trajectory length
    cutoff = 0

    # add trajectory index on the video or not
    # if you want, set it to True
    add_index = False

    ###############################################################################
    if len(sys.argv) < 2:
        sys.exit('Please secify input trxyt file\nExample: python make_image.py sample.trxyt')
    else:
        input_file = sys.argv[1]
        output_file = f'{input_file.strip().split("/")[-1].split(".trxyt")[0]}.tiff'
        print(f'Selected option: {resolution} resolution, cutoff={cutoff}, add_index={add_index}')
        resolution = int(resolution - (resolution%10))
        make_image_seqs(trxyt_file=input_file, output_dir=output_file, cutoff=cutoff, add_index=add_index, resolution=resolution)
