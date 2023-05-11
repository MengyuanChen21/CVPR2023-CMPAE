import shutil
import subprocess
import os
import os.path as osp
import argparse
import glob
# from mmcv import ProgressBar
from multiprocessing import Process
import random


def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "8 "
    command1 += '{0}/%06d.jpg'.format(dst)
    # print(command1)
    os.system(command1)


def multi_process(vid_list, video_path, out_dir):
    for vid_id in vid_list:
        name = os.path.join(video_path, vid_id)
        dst = os.path.join(out_dir, vid_id[:-4])
        if not os.path.exists(dst):
            os.makedirs(dst)
            extract_frames(name, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/LLP_dataset/frame')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/LLP_dataset/video')
    # parser.add_argument('--log_dir', dest='log_dir', type=str, default='log/extract_frames')
    args = parser.parse_args()
    # log_dir = args.log_dir
    vid_list = os.listdir(args.video_path)
    random.shuffle(vid_list)

    thread_number = 10
    vid_per_process = (len(vid_list) + thread_number - 1) // thread_number
    procs = list()
    for i in range(thread_number):
        proc = Process(target=multi_process, args=(vid_list[i * vid_per_process: (i + 1) * vid_per_process],
                                                   args.video_path, args.out_dir))
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()
