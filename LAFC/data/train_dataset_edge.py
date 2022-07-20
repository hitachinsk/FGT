import random

import pickle
import logging
import torch
import cv2
import os

from torch.utils.data.dataset import Dataset
import numpy as np
from skimage.feature import canny
from .util.STTN_mask import create_random_shape_with_random_motion
from cvbase import read_flow, flow2rgb
from .util.flow_utils import region_fill as rf
import imageio

logger = logging.getLogger('base')


class VideoBasedDataset(Dataset):
    def __init__(self, opt, dataInfo):
        self.opt = opt
        self.mode = opt['mode']
        self.sampleMethod = opt['sample']
        self.dataInfo = dataInfo
        self.flow_height, self.flow_width = dataInfo['flow']['flow_height'], dataInfo['flow']['flow_width']
        self.data_path = dataInfo['flow_path']
        self.frame_path = dataInfo['frame_path']
        self.train_list = os.listdir(self.data_path)
        self.name2length = self.dataInfo['name2len']
        self.require_edge = opt['use_edges']
        self.sigma = dataInfo['edge']['sigma']
        self.low_threshold = dataInfo['edge']['low_threshold']
        self.high_threshold = dataInfo['edge']['high_threshold']
        with open(self.name2length, 'rb') as f:
            self.name2len = pickle.load(f)
        self.norm = opt['norm']
        self.sequenceLen = self.opt['num_flows']
        self.flow_interval = self.opt['flow_interval']
        self.halfLen = self.sequenceLen // 2

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        try:
            item = self.load_item(idx)
        except:
            print('Loading error: ' + self.train_list[idx])
            item = self.load_item(0)
        return item

    def frameSample(self, flowLen):
        if self.sampleMethod == 'random':
            indices = [i for i in range(flowLen)]
            sampledIndices = random.sample(indices, self.sequenceLen)
        else:
            sampledIndices = []
            pivot = random.randint(0, flowLen - 1)
            for i in range(-self.halfLen, self.halfLen + 1):
                index = pivot + i * self.flow_interval
                if index < 0:
                    index = 0
                if index >= flowLen:
                    index = flowLen - 1
                sampledIndices.append(index)
        return sampledIndices

    def load_item(self, idx):
        info = {}
        video = self.train_list[idx]
        info['name'] = video
        if np.random.uniform(0, 1) > 0.5:
            direction = 'forward_flo'
        else:
            direction = 'backward_flo'
        flow_dir = os.path.join(self.data_path, video, direction)
        frame_dir = os.path.join(self.frame_path, video)
        flowLen = self.name2len[video] - 1
        assert flowLen > self.sequenceLen, 'Flow length {} is not enough'.format(flowLen)
        sampledIndices = self.frameSample(flowLen)
        candidateMasks = create_random_shape_with_random_motion(self.sequenceLen, 0.9, 1.1, 1,
                                                                10)
        flows, diffused_flows, masks = [], [], []
        current_frames, shift_frames = None, None
        mask_counter = 0
        for i in sampledIndices:
            flow = read_flow(os.path.join(flow_dir, '{:05d}.flo'.format(i)))
            mask = self.read_mask(candidateMasks[mask_counter], self.flow_height, self.flow_width)
            mask_counter += 1
            flow = self.flow_tf(flow, self.flow_height, self.flow_width)
            diffused_flow = self.diffusion_fill(flow, mask)
            flows.append(flow)
            masks.append(mask)
            diffused_flows.append(diffused_flow)
        targetIndex = sampledIndices[self.sequenceLen // 2]
        current_frames, shift_frames = self.read_frames(frame_dir, targetIndex, direction, self.flow_width,
                                                        self.flow_height)
        flow_gray, edge = self.load_edge(flows[self.halfLen])
        inputs = {'flows': flows, 'diffused_flows': diffused_flows, 'current_frame': current_frames,
                  'shift_frame': shift_frames, 'edges': edge, 'masks': masks, 'flow_gray': flow_gray}
        return self.to_tensor(inputs)

    def read_frames(self, frame_dir, index, direction, width, height):
        if direction == 'forward_flo':
            current_frame = os.path.join(frame_dir, '{:05d}.jpg'.format(index))
            shift_frame = os.path.join(frame_dir, '{:05d}.jpg'.format(index + 1))
        else:
            current_frame = os.path.join(frame_dir, '{:05d}.jpg'.format(index + 1))
            shift_frame = os.path.join(frame_dir, '{:05d}.jpg'.format(index))
        current_frame = imageio.imread(current_frame)
        shift_frame = imageio.imread(shift_frame)
        current_frame = cv2.resize(current_frame, (width, height), cv2.INTER_LINEAR)
        shift_frame = cv2.resize(shift_frame, (width, height), cv2.INTER_LINEAR)
        current_frame = current_frame / 255.
        shift_frame = shift_frame / 255.
        return current_frame, shift_frame

    def diffusion_fill(self, flow, mask):
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0] * (1 - mask), mask)
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1] * (1 - mask), mask)
        return flow_filled

    def flow_tf(self, flow, height, width):
        flow_shape = flow.shape
        flow_resized = cv2.resize(flow, (width, height), cv2.INTER_LINEAR)
        flow_resized[:, :, 0] *= (float(width) / float(flow_shape[1]))
        flow_resized[:, :, 1] *= (float(height) / float(flow_shape[0]))
        return flow_resized

    def read_mask(self, mask, height, width):
        mask = np.array(mask)
        mask = mask / 255.
        raw_mask = (mask > 0.5).astype(np.uint8)
        raw_mask = cv2.resize(raw_mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        return raw_mask

    def load_edge(self, flow):
        gray_flow = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** 0.5
        factor = gray_flow.max()
        gray_flow = gray_flow / factor
        flow_rgb = flow2rgb(flow)
        flow_gray = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
        return gray_flow, canny(flow_gray, sigma=self.sigma, mask=None, low_threshold=self.low_threshold,
                     high_threshold=self.high_threshold).astype(np.float)

    def to_tensor(self, data_list):
        """

        Args:
            data_list: a numpy.array list

        Returns: a torch.tensor list with the None entries removed

        """
        keys = list(data_list.keys())
        for key in keys:
            if data_list[key] is None or data_list[key] == []:
                data_list.pop(key)
            else:
                item = data_list[key]
                if not isinstance(item, list):
                    if len(item.shape) == 2:
                        item = item[:, :, np.newaxis]
                    item = torch.from_numpy(np.transpose(item, (2, 0, 1))).float()
                else:
                    item = np.stack(item, axis=0)
                    if len(item.shape) == 3:
                        item = item[:, :, :, np.newaxis]
                    item = torch.from_numpy(np.transpose(item, (3, 0, 1, 2))).float()
                data_list[key] = item
        return data_list

