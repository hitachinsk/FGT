import torch
import numpy as np
from .sobel2 import SobelLayer, SeperateSobelLayer
import torch.nn as nn
import torch.nn.functional as F


def image_warp(image, flow):
    '''
    image: 上一帧的图片,torch.Size([1, 3, 256, 256])
    flow: 光流, torch.Size([1, 2, 256, 256])
    final_grid:  torch.Size([1, 2, 256, 256])
    '''
    b, c, h, w = image.size()
    device = image.device
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)],
                     dim=1)  # normalize to [-1~1](from upper left to lower right
    flow = flow.permute(0, 2, 3,
                        1)  # if you wanna use grid_sample function, the channel(band) shape of show must be in the last dimension
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    grid = torch.cat((torch.from_numpy(X.astype('float32')).unsqueeze(0).unsqueeze(3),
                      torch.from_numpy(Y.astype('float32')).unsqueeze(0).unsqueeze(3)), 3).to(device)
    output = torch.nn.functional.grid_sample(image, grid + flow, mode='bilinear', padding_mode='zeros')
    return output


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = image_warp(flow_bw, flow_fw)  # wb(wf(x))
    flow_fw_warped = image_warp(flow_fw, flow_bw)  # wf(wb(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))
    flow_diff_bw = flow_bw + flow_fw_warped  # wb + wf(wb(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)  # |wb| + |wf(wb(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2
    occ_thresh_bw = alpha1 * mag_sq_bw + alpha2

    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()

    return fb_occ_fw, fb_occ_bw  # fb_occ_fw -> frame2 area occluded by frame1, fb_occ_bw -> frame1 area occluded by frame2


def rgb2gray(image):
    gray_image = image[:, 0] * 0.299 + image[:, 1] * 0.587 + 0.110 * image[:, 2]
    gray_image = gray_image.unsqueeze(1)
    return gray_image


def ternary_transform(image, max_distance=1):
    device = image.device
    patch_size = 2 * max_distance + 1
    intensities = rgb2gray(image) * 255
    out_channels = patch_size * patch_size
    w = np.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
    weights = torch.from_numpy(w).float().to(device)
    patches = F.conv2d(intensities, weights, stride=1, padding=1)
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
    return transf_norm


def hamming_distance(t1, t2):
    dist = torch.square(t1 - t2)
    dist_norm = dist / (0.1 + dist)
    dist_sum = torch.sum(dist_norm, dim=1, keepdim=True)
    return dist_sum


def create_mask(mask, paddings):
    """
    padding: [[top, bottom], [left, right]]
    """
    shape = mask.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])

    mask2d = F.pad(inner, pad=[paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]])  # mask最外边一圈都pad成0了
    mask3d = mask2d.unsqueeze(0)
    mask4d = mask3d.unsqueeze(0).repeat(shape[0], 1, 1, 1)
    return mask4d.detach()


def ternary_loss2(frame1, warp_frame21, confMask, masks, max_distance=1):
    """

    Args:
        frame1: torch tensor, with shape [b * t, c, h, w]
        warp_frame21: torch tensor, with shape [b * t, c, h, w]
        confMask: confidence mask, with shape [b * t, c, h, w]
        masks: torch tensor, with shape [b * t, c, h, w]
        max_distance: maximum distance.

    Returns: ternary loss

    """
    t1 = ternary_transform(frame1)
    t21 = ternary_transform(warp_frame21)
    dist = hamming_distance(t1, t21)  # 近似求解，其实利用了mask区域和外界边缘交叉的那一部分像素
    loss = torch.mean(dist * confMask * masks) / torch.mean(masks)
    return loss


def gradient_loss(frame1, frame2, confMask):
    device = frame1.device
    frame1_edge = SobelLayer(device)(frame1)
    frame2_edge = SobelLayer(device)(frame2)
    loss = torch.sum(torch.abs(frame1_edge * confMask - frame2_edge * confMask)) / (torch.sum(confMask) + 1)  # escape divide 0
    return loss


def seperate_gradient_loss(frame1, warp_frame21, confMask):
    device = frame1.device
    mask_x = create_mask(frame1, [[0, 0], [1, 1]]).to(device)
    mask_y = create_mask(frame1, [[1, 1], [0, 0]]).to(device)
    gradient_mask = torch.cat([mask_x, mask_y], dim=1).repeat(1, 3, 1, 1)
    frame1_edge = SeperateSobelLayer(device)(frame1)
    warp_frame21_edge = SeperateSobelLayer(device)(warp_frame21)
    loss = nn.L1Loss()(frame1_edge * confMask * gradient_mask, warp_frame21_edge * confMask * gradient_mask)
    return loss
