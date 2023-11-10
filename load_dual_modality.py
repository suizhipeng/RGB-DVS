from logging import raiseExceptions
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

def ev2image(evs, sensor_size):  # 事件数据进行可视化，用红色和蓝色表示 sensor_size=(W, H)
    if evs.ndim == 1:
        xs, ys, ps = evs['x'], evs['y'], evs['p']
    elif evs.ndim == 2:
        xs, ys, ps = evs[:, 0], evs[:, 1], evs[:, 2]
    else:
        print('Error')
        exit()
    W, H = sensor_size
    board = np.zeros((H, W, 3), dtype=np.uint8)
    board[ys[ps == 1].astype(np.int32), xs[ps == 1].astype(np.int32), 2] = 255  # 红色，极性为1
    board[ys[ps == 0].astype(np.int32), xs[ps == 0].astype(np.int32), 0] = 255  # 蓝色，极性为0
    return board

def superpose_event_to_frame(im_frame, ev_frame):  # 事件数据可视化在RGB图像上，直接修改event frame对应像素的通道（相比逐事件修改要更高效）
    frame_event_hybrid = im_frame.copy()
    frame_event_hybrid[ev_frame[:, :, 0] > 128, :] = ev_frame[ev_frame[:, :, 0] > 128, :]
    frame_event_hybrid[ev_frame[:, :, 2] > 128, :] = ev_frame[ev_frame[:, :, 2] > 128, :]
    return frame_event_hybrid

if __name__ == '__main__':
    root_path = ""

    # 进行映射
    rgb_list = glob.glob(os.path.join(root_path, 'raw_frame/*.png'))  # len(rgb_list) is 500
    rgb_list.sort()
    raw_events_list = glob.glob(os.path.join(root_path, 'time_split/*.npy'))  # len(raw_events_list) is 499
    raw_events_list.sort()

    direct_align = os.path.join(root_path, "direct_align")
    if not os.path.exists(direct_align):
        os.mkdir(direct_align)

    alpha = 0.3
    # length_list = []

    H = np.load("")
    for idx in range(len(raw_events_list)):
        print(idx)
        img = cv2.imread(rgb_list[idx+1])
        raw_events = np.load(raw_events_list[idx])
        # length_list.append(len(raw_events))
        # continue
        
        event = ev2image(raw_events, (1280, 720))
        wrap_event = cv2.warpPerspective(event, H, (1440, 1080))
        frame_event_hybrid = superpose_event_to_frame(img, wrap_event)
        cv2.imshow('window', frame_event_hybrid)
        cv2.waitKey(50)

        # layer = img.copy()
        # cv2.addWeighted(frame_event_hybrid, alpha, layer, 1 - alpha, 0, layer)
        # cv2.imshow('layer', layer)
        # cv2.waitKey(50)
        # cv2.imwrite(os.path.join(direct_align, '{0:0>6d}.png'.format(idx+1)), layer)

    # plt.hist(length_list, bins=10, rwidth=0.7)
    # plt.show()
