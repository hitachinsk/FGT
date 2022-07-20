import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .fbConsistencyCheck import image_warp


class FlowWarpingLoss(nn.Module):
    def __init__(self, metric):
        super(FlowWarpingLoss, self).__init__()
        self.metric = metric

    def warp(self, x, flow):
        """

        Args:
            x: torch tensor with shape [b, c, h, w], the x can be 3 (for rgb frame) or 2 (for optical flow)
            flow: torch tensor with shape [b, 2, h, w]

        Returns: the warped x (can be an image or an optical flow)

        """
        h, w = x.shape[2:]
        device = x.device
        # normalize the flow to [-1~1]
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1) / 2), flow[:, 1:2, :, :] / ((h - 1) / 2)], dim=1)
        flow = flow.permute(0, 2, 3, 1)  # change to [b, h, w, c]
        # generate meshgrid
        x_idx = np.linspace(-1, 1, w)
        y_idx = np.linspace(-1, 1, h)
        X_idx, Y_idx = np.meshgrid(x_idx, y_idx)
        grid = torch.cat((torch.from_numpy(X_idx.astype('float32')).unsqueeze(0).unsqueeze(3),
                          torch.from_numpy(Y_idx.astype('float32')).unsqueeze(0).unsqueeze(3)), 3).to(device)
        output = torch.nn.functional.grid_sample(x, grid + flow, mode='bilinear', padding_mode='zeros')
        return output

    def __call__(self, x, y, flow, mask):
        """
        image/flow warping, only support the single image/flow warping
        Args:
            x: Can be optical flow or image with shape [b, c, h, w], c can be 2 or 3
            y: The ground truth of x (can be the extracted optical flow or image)
            flow: The flow used to warp x, whose shape is [b, 2, h, w]
            mask: The mask which indicates the hole of x, which must be [b, 1, h, w]

        Returns: the warped image/optical flow

        """
        warped_x = self.warp(x, flow)
        loss = self.metric(warped_x * mask, y * mask)
        return loss


class TVLoss():
    # shift one pixel to get difference ( for both x and y direction)
    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, x):
        loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
            torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return loss


class WarpLoss(nn.Module):
    def __init__(self):
        super(WarpLoss, self).__init__()
        self.metric = nn.L1Loss()

    def forward(self, flow, mask, img1, img2):
        """

        Args:
            flow: flow indicates the motion from img1 to img2
            mask: mask corresponds to img1
            img1: frame 1
            img2: frame t+1

        Returns: warp loss from img2 to img1

        """
        img2_warped = image_warp(img2, flow)
        loss = self.metric(img2_warped * mask, img1 * mask)
        return loss


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


# Some losses related to optical flows
# From Unflow: https://github.com/simonmeister/UnFlow
def fbLoss(forward_flow, backward_flow, forward_gt_flow, backward_gt_flow, fb_loss_weight, image_warp_loss_weight=0,
           occ_weight=0, beta=255, first_image=None, second_image=None):
    """
    calculate the forward-backward consistency loss and the related image warp loss
    Args:
        forward_flow: torch tensor, with shape [b, c, h, w]
        backward_flow: torch tensor, with shape [b, c, h, w]
        forward_gt_flow: the ground truth of the forward flow (used for occlusion calculation)
        backward_gt_flow: the ground truth of the backward flow (used for occlusion calculation)
        fb_loss_weight: loss weight for forward-backward consistency check between two frames
        image_warp_loss_weight: loss weight for image warping
        occ_weight: loss weight for occlusion area (serve as a punishment for image warp loss)
        beta: 255 by default, according to the original loss codes in unflow
        first_image: the previous image (extraction for the optical flows)
        second_image: the later image (extraction for the optical flows)
        Note: forward and backward flow should be extracted from the same image pair
    Returns: forward backward consistency loss between forward and backward flow

    """
    mask_fw = create_outgoing_mask(forward_flow).float()
    mask_bw = create_outgoing_mask(backward_flow).float()

    # forward warp backward flow and backward forward flow to calculate the cycle consistency
    forward_flow_warped = image_warp(forward_flow, backward_gt_flow)
    forward_flow_warped_gt = image_warp(forward_gt_flow, backward_gt_flow)
    backward_flow_warped = image_warp(backward_flow, forward_gt_flow)
    backward_flow_warped_gt = image_warp(backward_gt_flow, forward_gt_flow)
    flow_diff_fw = backward_flow_warped + forward_flow
    flow_diff_fw_gt = backward_flow_warped_gt + forward_gt_flow
    flow_diff_bw = backward_flow + forward_flow_warped
    flow_diff_bw_gt = backward_gt_flow + forward_flow_warped_gt

    # occlusion calculation
    mag_sq_fw = length_sq(forward_gt_flow) + length_sq(backward_flow_warped_gt)
    mag_sq_bw = length_sq(backward_gt_flow) + length_sq(forward_flow_warped_gt)
    occ_thresh_fw = 0.01 * mag_sq_fw + 0.5
    occ_thresh_bw = 0.01 * mag_sq_bw + 0.5

    fb_occ_fw = (length_sq(flow_diff_fw_gt) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw_gt) > occ_thresh_bw).float()

    mask_fw *= (1 - fb_occ_fw)
    mask_bw *= (1 - fb_occ_bw)

    occ_fw = 1 - mask_fw
    occ_bw = 1 - mask_bw

    if image_warp_loss_weight != 0:
        # warp images
        second_image_warped = image_warp(second_image, forward_flow)  # frame 2 -> 1
        first_image_warped = image_warp(first_image, backward_flow)  # frame 1 -> 2
        im_diff_fw = first_image - second_image_warped
        im_diff_bw = second_image - first_image_warped
        # calculate the image warp loss based on the occlusion regions calculated by forward and backward flows (gt)
        occ_loss = occ_weight * (charbonnier_loss(occ_fw) + charbonnier_loss(occ_bw))
        image_warp_loss = image_warp_loss_weight * (
                charbonnier_loss(im_diff_fw, mask_fw, beta=beta) + charbonnier_loss(im_diff_bw, mask_bw,
                                                                                    beta=beta)) + occ_loss
    else:
        image_warp_loss = 0
    fb_loss = fb_loss_weight * (charbonnier_loss(flow_diff_fw, mask_fw) + charbonnier_loss(flow_diff_bw, mask_bw))
    return fb_loss + image_warp_loss


def length_sq(x):
    return torch.sum(torch.square(x), 1, keepdim=True)


def smoothness_loss(flow, cmask):
    delta_u, delta_v, mask = smoothness_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def smoothness_deltas(flow):
    """
    flow: [b, c, h, w]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask


def second_order_loss(flow, cmask):
    delta_u, delta_v, mask = second_order_deltas(flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    x: a tensor of shape [b, c, h, w]
    mask: a mask of shape [b, mc, h, w], where mask channels must be either 1 or the same as
    the number of channels of x. Entries should be 0 or 1
    return: loss
    """
    b, c, h, w = x.shape
    norm = b * c * h * w
    error = torch.pow(torch.square(x * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm


def second_order_deltas(flow):
    """
    consider the single flow first
    flow shape: [b, c, h, w]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=1)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=1)
    return delta_u, delta_v, mask


def create_mask(tensor, paddings):
    """
    tensor shape: [b, c, h, w]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = tensor.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [paddings[1][0], paddings[1][1], paddings[0][0], paddings[0][1]]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()


def create_outgoing_mask(flow):
    """
    Computes a mask that is zero at all positions where the flow would carry a pixel over the image boundary
    For such pixels, it's invalid to calculate the flow losses
    Args:
        flow: torch tensor: with shape [b, 2, h, w]

    Returns: a mask, 1 indicates in-boundary pixels, with shape [b, 1, h, w]

    """
    b, c, h, w = flow.shape

    grid_x = torch.reshape(torch.arange(0, w), [1, 1, w])
    grid_x = grid_x.repeat(b, h, 1).float()
    grid_y = torch.reshape(torch.arange(0, h), [1, h, 1])
    grid_y = grid_y.repeat(b, 1, w).float()

    grid_x = grid_x.to(flow.device)
    grid_y = grid_y.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)  # [b, h, w]
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v
    inside_x = torch.logical_and(pos_x <= w - 1, pos_x >= 0)
    inside_y = torch.logical_and(pos_y <= h - 1, pos_y >= 0)
    inside = torch.logical_and(inside_x, inside_y)
    if len(inside.shape) == 3:
        inside = inside.unsqueeze(1)
    return inside
