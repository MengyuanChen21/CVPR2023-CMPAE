import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import argparse
from multiprocessing import Process
import random


def extract(sound_list, video_path, save_path):
    # process_videos = list()
    # exist_lis = os.listdir(save_path)
    for audio_id in sound_list:
        name = os.path.join(video_path, audio_id)
        audio_name = audio_id[:-4] + '.wav'
        if os.path.exists(os.path.join(save_path, audio_name)):
            print("already exist!")
            continue
        try:
            video = VideoFileClip(name)
            audio = video.audio
            audio.write_audiofile(os.path.join(save_path, audio_name), fps=16000)
        except:
            print("cannot load ", name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/LLP_dataset/audio')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/LLP_dataset/video')
    # parser.add_argument('--log_dir', dest='log_dir', type=str, default='log/extract_audio')
    args = parser.parse_args()

    sound_list = os.listdir(args.video_path)
    random.shuffle(sound_list)
    threads = 10
    number_per_thread = (len(sound_list) + threads - 1) // threads
    procs = list()

    for i in range(threads):
        proc = Process(target=extract, args=(sound_list[i * number_per_thread: (i + 1) * number_per_thread],
                                             args.video_path, args.out_dir))
        proc.start()
        procs.append(proc)
    for p in procs:
        p.join()

