import random
import pickle

import logging
import torch
import cv2
import os

from torch.utils.data.dataset import Dataset
import numpy as np
import cvbase
from .util.STTN_mask import create_random_shape_with_random_motion
import imageio
from .util.flow_utils import region_fill as rf

logger = logging.getLogger('base')


class VideoBasedDataset(Dataset):
    def __init__(self, opt, dataInfo):
        self.opt = opt
        self.sampleMethod = opt['sample']
        self.dataInfo = dataInfo
        self.height, self.width = self.opt['input_resolution']
        self.frame_path = dataInfo['frame_path']
        self.flow_path = dataInfo['flow_path']  # The path of the optical flows
        self.train_list = os.listdir(self.frame_path)
        self.name2length = self.dataInfo['name2len']
        with open(self.name2length, 'rb') as f:
            self.name2length = pickle.load(f)
        self.sequenceLen = self.opt['num_frames']
        self.flow2rgb = opt['flow2rgb']  # whether to change flow to rgb domain
        self.flow_direction = opt[
            'flow_direction']  # The direction must be in ['for', 'back', 'bi'], indicating forward, backward and bidirectional flows

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        try:
            item = self.load_item(idx)
        except:
            print('Loading error: ' + self.train_list[idx])
            item = self.load_item(0)
        return item

    def frameSample(self, frameLen, sequenceLen):
        if self.sampleMethod == 'random':
            indices = [i for i in range(frameLen)]
            sampleIndices = random.sample(indices, sequenceLen)
        elif self.sampleMethod == 'seq':
            pivot = random.randint(0, sequenceLen - 1 - frameLen)
            sampleIndices = [i for i in range(pivot, pivot + frameLen)]
        else:
            raise ValueError('Cannot determine the sample method {}'.format(self.sampleMethod))
        return sampleIndices

    def load_item(self, idx):
        video = self.train_list[idx]
        frame_dir = os.path.join(self.frame_path, video)
        forward_flow_dir = os.path.join(self.flow_path, video, 'forward_flo')
        backward_flow_dir = os.path.join(self.flow_path, video, 'backward_flo')
        frameLen = self.name2length[video]
        flowLen = frameLen - 1
        assert frameLen > self.sequenceLen, 'Frame length {} is less than sequence length'.format(frameLen)
        sampledIndices = self.frameSample(frameLen, self.sequenceLen)

        # generate random masks for these sampled frames
        candidateMasks = create_random_shape_with_random_motion(frameLen, 0.9, 1.1, 1, 10)

        # read the frames and masks
        frames, masks, forward_flows, backward_flows = [], [], [], []
        for i in range(len(sampledIndices)):
            frame = self.read_frame(os.path.join(frame_dir, '{:05d}.jpg'.format(sampledIndices[i])), self.height,
                                    self.width)
            mask = self.read_mask(candidateMasks[sampledIndices[i]], self.height, self.width)
            frames.append(frame)
            masks.append(mask)
            if self.flow_direction == 'for':
                forward_flow = self.read_forward_flow(forward_flow_dir, sampledIndices[i], flowLen)
                forward_flow = self.diffusion_flow(forward_flow, mask)
                forward_flows.append(forward_flow)
            elif self.flow_direction == 'back':
                backward_flow = self.read_backward_flow(backward_flow_dir, sampledIndices[i])
                backward_flow = self.diffusion_flow(backward_flow, mask)
                backward_flows.append(backward_flow)
            elif self.flow_direction == 'bi':
                forward_flow = self.read_forward_flow(forward_flow_dir, sampledIndices[i], flowLen)
                forward_flow = self.diffusion_flow(forward_flow, mask)
                forward_flows.append(forward_flow)
                backward_flow = self.read_backward_flow(backward_flow_dir, sampledIndices[i])
                backward_flow = self.diffusion_flow(backward_flow, mask)
                backward_flows.append(backward_flow)
            else:
                raise ValueError('Unknown flow direction mode: {}'.format(self.flow_direction))
        inputs = {'frames': frames, 'masks': masks, 'forward_flo': forward_flows, 'backward_flo': backward_flows}
        inputs = self.to_tensor(inputs)
        inputs['frames'] = (inputs['frames'] / 255.) * 2 - 1
        return inputs

    def diffusion_flow(self, flow, mask):
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0] * (1 - mask), mask)
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1] * (1 - mask), mask)
        return flow_filled

    def read_frame(self, path, height, width):
        frame = imageio.imread(path)
        frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
        return frame

    def read_mask(self, mask, height, width):
        mask = np.array(mask)
        mask = mask / 255.
        raw_mask = (mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        return raw_mask

    def read_forward_flow(self, forward_flow_dir, sampledIndex, flowLen):
        if sampledIndex >= flowLen:
            sampledIndex = flowLen - 1
        flow = cvbase.read_flow(os.path.join(forward_flow_dir, '{:05d}.flo'.format(sampledIndex)))
        height, width = flow.shape[:2]
        flow = cv2.resize(flow, (self.width, self.height), cv2.INTER_LINEAR)
        flow[:, :, 0] = flow[:, :, 0] / width * self.width
        flow[:, :, 1] = flow[:, :, 1] / height * self.height
        return flow

    def read_backward_flow(self, backward_flow_dir, sampledIndex):
        if sampledIndex == 0:
            sampledIndex = 0
        else:
            sampledIndex -= 1
        flow = cvbase.read_flow(os.path.join(backward_flow_dir, '{:05d}.flo'.format(sampledIndex)))
        height, width = flow.shape[:2]
        flow = cv2.resize(flow, (self.width, self.height), cv2.INTER_LINEAR)
        flow[:, :, 0] = flow[:, :, 0] / width * self.width
        flow[:, :, 1] = flow[:, :, 1] / height * self.height
        return flow

    def to_tensor(self, data_list):
        """

        Args:
            data_list: A list contains multiple numpy arrays

        Returns: The stacked tensor list

        """
        keys = list(data_list.keys())
        for key in keys:
            if data_list[key] is None or data_list[key] == []:
                data_list.pop(key)
            else:
                item = data_list[key]
                if not isinstance(item, list):
                    item = torch.from_numpy(np.transpose(item, (2, 0, 1))).float()  # [c, h, w]
                else:
                    item = np.stack(item, axis=0)
                    if len(item.shape) == 3:  # [t, h, w]
                        item = item[:, :, :, np.newaxis]
                    item = torch.from_numpy(np.transpose(item, (0, 3, 1, 2))).float()  # [t, c, h, w]
                data_list[key] = item
        return data_list

