# coding=utf-8
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import os
import cv2
import glob
import copy
import numpy as np
import torch
from PIL import Image
import scipy.ndimage
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from RAFT import utils
from RAFT import RAFT

import utils.region_fill as rf
from torchvision.transforms import ToTensor
import time


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def gradient_mask(mask):  # 产生梯度的mask

    gradient_mask = np.logical_or.reduce((mask,
                                          np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)),
                                                         axis=0),
                                          np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)),
                                                         axis=1)))

    return gradient_mask


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(args, model, vid, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    create_dir(os.path.join(args.outroot, vid, mode + '_flo'))
    # create_dir(os.path.join(args.outroot, vid, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            # Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            # flow_img = utils.flow_viz.flow_to_image(flow)
            # flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            # flow_img.save(os.path.join(args.outroot, vid, 'flow', mode + '_png', '%05d.png'%i))
            utils.frame_utils.writeFlow(os.path.join(args.outroot, vid, mode + '_flo', '%05d.flo' % i), flow)


def main(args):
    # Flow model.
    RAFT_model = initialize_RAFT(args)

    videos = os.listdir(args.path)
    videoLen = len(videos)
    try:
        exceptList = os.listdir(args.expdir)
    except:
        exceptList = []
    v = 0
    for vid in videos:
        v += 1
        print('[{}]/[{}] Video {} is being processed'.format(v, len(videos), vid))
        if vid in exceptList:
            print('Video: {} skipped'.format(vid))
            continue
        # Loads frames.
        filename_list = glob.glob(os.path.join(args.path, vid, '*.png')) + \
                        glob.glob(os.path.join(args.path, vid, '*.jpg'))

        # Obtains imgH, imgW and nFrame.
        imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
        nFrame = len(filename_list)
        print('images are loaded')

        # Loads video.
        video = []
        for filename in sorted(filename_list):
            print(filename)
            img = np.array(Image.open(filename))
            if args.width != 0 and args.height != 0:
                img = cv2.resize(img, (args.width, args.height), cv2.INTER_LINEAR)
            video.append(torch.from_numpy(img.astype(np.uint8)).permute(2, 0, 1).float())

        video = torch.stack(video, dim=0)
        video = video.to('cuda')

        # Calcutes the corrupted flow.
        start = time.time()
        calculate_flow(args, RAFT_model, vid, video, 'forward')
        calculate_flow(args, RAFT_model, vid, video, 'backward')
        end = time.time()
        sumTime = end - start
        print('{}/{}, video {} is finished. {} frames takes {}s, {}s/frame.'.format(v, videoLen, vid, nFrame, sumTime,
                                                                                    sumTime / (2 * nFrame)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # flow basic setting
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--expdir', type=str)
    parser.add_argument('--outroot', required=True, type=str)
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument('--height', type=int, default=256)

    # RAFT
    parser.add_argument('--model', default='../weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    main(args)

