import shutil
import sys
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
import argparse
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import transforms as TF
import utils
import torchvision
from mmcv import ProgressBar
import random


C, H, W = 3, 112, 112


def extract_feats(params, model, load_img):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    video_list = os.listdir(params['video_path'])
    random.shuffle(video_list)
    nn = 0
    process_videos = list()
    pb = ProgressBar(len(video_list))
    pb.start()
    for video in video_list:
        outfile = os.path.join(dir_fc, video + '.npy')
        if os.path.exists(outfile):
            pb.update()
            continue
        nn = nn + 1
        dst = video

        image_list = sorted(glob.glob(os.path.join(params['video_path'], dst, '*.jpg')))
        number = len(image_list)
        samples = np.round(np.linspace(
            0, number - 1, params['n_frame_steps']))
        try:
            image_list = [image_list[min(int(sample), number - 1)] for sample in samples]
        except:
            pb.update()
            continue
        images = torch.zeros((len(image_list)//8, C, 8, H, W))
        i = 0
        for iImg in range(len(image_list)):

            ii = i // 8
            img = load_img(image_list[iImg])
            images[ii, :, i % 8, :, :] = img
            i += 1

        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        np.save(outfile, img_feats)
        # cleanup
        #shutil.rmtree(dst)
        # print(nn)
        process_videos.append(video)
        pb.update()

    # os.makedirs(params["log_dir"], exist_ok=True)
    # wfile = "extract" + str(len([f for f in os.listdir(params["log_dir"]) if 'extract' in f])) + ".txt"
    # with open(osp.join(params["log_dir"], wfile), 'w', encoding='utf-8') as f:
    #     f.write("\n".join(process_videos))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='9',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/r2plus1d_18', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/LLP_dataset/frame', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='r2plus1d_18',
                        help='the CNN model you want to use to extract_feats')

    # parser.add_argument('--log_dir', dest='log_dir', type=str, default='log/extract_3D_feat')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'r2plus1d_18':
        model = models.video.r2plus1d_18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        for param in model.parameters():
            param.requires_grad = False
        T, C, H, W = 8, 3, 112, 112
        load_img = utils.LoadTransformImage()

    else:
        print("doesn't support %s" % (params['model']))

    model = nn.DataParallel(model)
    model = model.cuda()
    extract_feats(params, model, load_img)
