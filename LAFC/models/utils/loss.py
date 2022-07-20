import os
import sys

import torch
import torch.nn as nn
import numpy as np


class AlignLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, frames, masks, aligned_vs, aligned_rs):
        """

        :param frames: The original frames(GT)
        :param masks: Original masks
        :param aligned_vs: aligned visibility map from reference frame(List: B, C, T, H, W)
        :param aligned_rs: aligned reference frames(List: B, C, T, H, W)
        :return:
        """
        try:
            B, C, T, H, W = frames.shape
        except ValueError:
            frames = frames.unsqueeze(2)
            masks = masks.unsqueeze(2)
        B, C, T, H, W = frames.shape
        loss = 0
        for i in range(T):
            frame = frames[:, :, i]
            mask = masks[:, :, i]
            aligned_v = aligned_vs[i]
            aligned_r = aligned_rs[i]
            loss += self._singleFrameAlignLoss(frame, mask, aligned_v, aligned_r)
        return loss

    def _singleFrameAlignLoss(self, targetFrame, targetMask, aligned_v, aligned_r):
        """

        :param targetFrame: targetFrame to be aligned-> B, C, H, W
        :param targetMask: the mask of target frames
        :param aligned_v: aligned visibility map from reference frame
        :param aligned_r: aligned reference frame-> B, C, T, H, W
        :return:
        """
        targetVisibility = 1. - targetMask
        targetVisibility = targetVisibility.unsqueeze(2)
        targetFrame = targetFrame.unsqueeze(2)
        visibility_map = targetVisibility * aligned_v
        target_visibility = visibility_map * targetFrame
        reference_visibility = visibility_map * aligned_r
        loss = 0
        for i in range(aligned_r.shape[2]):
            loss += self.loss_fn(target_visibility[:, :, i], reference_visibility[:, :, i])
        return loss


class HoleVisibleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, outputs, masks, GTs, c_masks):
        try:
            B, C, T, H, W = outputs.shape
        except ValueError:
            outputs = outputs.unsqueeze(2)
            masks = masks.unsqueeze(2)
            GTs = GTs.unsqueeze(2)
            c_masks = c_masks.unsqueeze(2)
        B, C, T, H, W = outputs.shape
        loss = 0
        for i in range(T):
            loss += self._singleFrameHoleVisibleLoss(outputs[:, :, i], masks[:, :, i], c_masks[:, :, i], GTs[:, :, i])
        return loss

    def _singleFrameHoleVisibleLoss(self, targetFrame, targetMask, c_mask, GT):
        return self.loss_fn(targetMask * c_mask * targetFrame, targetMask * c_mask * GT)


class HoleInvisibleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, outputs, masks, GTs, c_masks):
        try:
            B, C, T, H, W = outputs.shape
        except ValueError:
            outputs = outputs.unsqueeze(2)
            masks = masks.unsqueeze(2)
            GTs = GTs.unsqueeze(2)
            c_masks = c_masks.unsqueeze(2)
        B, C, T, H, W = outputs.shape
        loss = 0
        for i in range(T):
            loss += self._singleFrameHoleInvisibleLoss(outputs[:, :, i], masks[:, :, i], c_masks[:, :, i], GTs[:, :, i])
        return loss

    def _singleFrameHoleInvisibleLoss(self, targetFrame, targetMask, c_mask, GT):
        return self.loss_fn(targetMask * (1. - c_mask) * targetFrame, targetMask * (1. - c_mask) * GT)


class NonHoleLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, outputs, masks, GTs):
        try:
            B, C, T, H, W = outputs.shape
        except ValueError:
            outputs = outputs.unsqueeze(2)
            masks = masks.unsqueeze(2)
            GTs = GTs.unsqueeze(2)
        B, C, T, H, W = outputs.shape
        loss = 0
        for i in range(T):
            loss += self._singleNonHoleLoss(outputs[:, :, i], masks[:, :, i], GTs[:, :, i])
        return loss

    def _singleNonHoleLoss(self, targetFrame, targetMask, GT):
        return self.loss_fn((1. - targetMask) * targetFrame, (1. - targetMask) * GT)


class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, model_output, target, mask):
        outputs = model_output
        targets = target
        if self.masked:
            masks = mask
            return self.loss_fn(outputs * masks, targets * masks)  # L1 loss in masked region
        else:
            return self.loss_fn(outputs, targets)  # L1 loss in the whole region


class VGGLoss(nn.Module):
    def __init__(self, vgg):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg = vgg

    def vgg_loss(self, output, target):
        output_feature = self.vgg(output)
        target_feature = self.vgg(target)
        loss = (
                self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
                + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
                + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, data_input, model_output):
        targets = data_input
        outputs = model_output
        mean_image_loss = self.vgg_loss(outputs, targets)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, vgg, original_channel_norm=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg = vgg
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, data_input, model_output):
        targets = data_input
        outputs = model_output
        mean_image_loss = self.style_loss(outputs, targets)
        return mean_image_loss


class L1LossMaskedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, mask):
        masked = 1 - mask  # 默认missing region的mask值为0，原有区域为1
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        masked = 1 - mask
        l2_sum = self.l2(x * masked, y * masked)
        return l2_sum / torch.sum(masked)


class ImcompleteVideoReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        imcomplete_video = model_output['imcomplete_video']
        targets = data_input['targets']
        down_sampled_targets = nn.functional.interpolate(
            targets.transpose(1, 2), scale_factor=[1, 0.5, 0.5])

        masks = data_input['masks']
        down_sampled_masks = nn.functional.interpolate(
            masks.transpose(1, 2), scale_factor=[1, 0.5, 0.5])
        return self.loss_fn(
            imcomplete_video, down_sampled_targets,
            down_sampled_masks
        )


class CompleteFramesReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        masks = data_input['masks']
        return self.loss_fn(outputs, targets, masks)


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
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


# # From https://github.com/phoenix104104/fast_blind_video_consistency
# class TemporalWarpingLoss(nn.Module):
#     def __init__(self, opts, flownet_checkpoint_path=None, alpha=50):
#         super().__init__()
#         self.loss_fn = L1LossMaskedMean()
#         self.alpha = alpha
#         self.opts = opts
#
#         assert flownet_checkpoint_path is not None, "Flownet2 pretrained models must be provided"
#
#         self.flownet_checkpoint_path = flownet_checkpoint_path
#         raise NotImplementedError
#
#     def get_flownet_checkpoint_path(self):
#         return self.flownet_checkpoint_path
#
#     def _flownetwrapper(self):
#         Flownet = FlowNet2(self.opts, requires_grad=False)
#         Flownet2_ckpt = torch.load(self.flownet_checkpoint_path)
#         Flownet.load_state_dict(Flownet2_ckpt['state_dict'])
#         Flownet.to(device)
#         Flownet.exal()
#         return Flownet
#
#     def _setup(self):
#         self.flownet = self._flownetwrapper()
#
#     def _get_non_occlusuib_mask(self, targets, warped_targets):
#         non_occlusion_masks = torch.exp(
#             -self.alpha * torch.sum(targets[:, 1:] - warped_targets, dim=2).pow(2)
#         ).unsqueeze(2)
#         return non_occlusion_masks
#
#     def _get_loss(self, outputs, warped_outputs, non_occlusion_masks, masks):
#         return self.loss_fn(
#             outputs[:, 1:] * non_occlusion_masks,
#             warped_outputs * non_occlusion_masks,
#             masks[:, 1:]
#         )
#
#     def forward(self, data_input, model_output):
#         if self.flownet is None:
#             self._setup()
#
#         targets = data_input['targets'].to(device)
#         outputs = model_output['outputs'].to(device)
#         flows = self.flownet.infer_video(targets).to(device)
#
#         from utils.flow_utils import warp_optical_flow
#         warped_targets = warp_optical_flow(targets[:, :-1], -flows).detach()
#         warped_outputs = warp_optical_flow(outputs[:, :-1], -flows).detach()
#         non_occlusion_masks = self._get_non_occlusion_mask(targets, warped_targets)
#
#         # model_output is passed by name and dictionary is mutable
#         # These values are sent to trainer for visualization
#         model_output['warped_outputs'] = warped_outputs[0]
#         model_output['warped_targets'] = warped_targets[0]
#         model_output['non_occlusion_masks'] = non_occlusion_masks[0]
#         from utils.flow_utils import flow_to_image
#         flow_imgs = []
#         for flow in flows[0]:
#             flow_img = flow_to_image(flow.cpu().permute(1, 2, 0).detach().numpy()).transpose(2, 0, 1)
#             flow_imgs.append(torch.Tensor(flow_img))
#         model_output['flow_imgs'] = flow_imgs
#
#         masks = data_input['masks'].to(device)
#         return self._get_loss(outputs, warped_outputs, non_occlusion_masks, masks)
#
#
# class TemporalWarpingError(TemporalWarpingLoss):
#     def __init__(self, flownet_checkpoint_path, alpha=50):
#         super().__init__(flownet_checkpoint_path, alpha)
#         self.loss_fn = L2LossMaskedMean(reduction='none')
#
#     def _get_loss(self, outputs, warped_outputs, non_occlusion_masks, masks):
#         # See https://arxiv.org/pdf/1808.00449.pdf 4.3
#         # The sum of non_occlusion_masks is different for each video,
#         # So the batch dim is kept
#         loss = self.loss_fn(
#             outputs[:, 1:] * non_occlusion_masks,
#             warped_outputs * non_occlusion_masks,
#             masks[:, 1:]
#         ).sum(1).sum(1).sum(1).sum(1)
#
#         loss = loss / non_occlusion_masks.sum(1).sum(1).sum(1).sum(1)
#         return loss.sum()


class ValidLoss(nn.Module):
    def __init__(self):
        super(ValidLoss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, model_output, target, mk):
        outputs = model_output
        targets = target
        return self.loss_fn(outputs * (1 - mk), targets * (1 - mk))  # L1 loss in masked region



class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, mask_input, model_output):
        # View 3D data as 2D
        outputs = model_output

        if len(mask_input.shape) == 4:
            mask_input = mask_input.unsqueeze(2)
        if len(outputs.shape) == 4:
            outputs = outputs.unsqueeze(2)

        outputs = outputs.permute((0, 2, 1, 3, 4)).contiguous()
        masks = mask_input.permute((0, 2, 1, 3, 4)).contiguous()

        B, L, C, H, W = outputs.shape
        x = outputs.view([B * L, C, H, W])

        masks = masks.view([B * L, -1])
        mask_areas = masks.sum(dim=1)

        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum(1).sum(1).sum(1)  # 差分是为了求梯度，本质上还是梯度平方和
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum(1).sum(1).sum(1)
        return ((h_tv + w_tv) / mask_areas).mean()


# for debug
def show_images(image, name):
    import cv2
    import numpy as np
    image = np.array(image)
    image[image > 0.5] = 255.
    image = image.transpose((1, 2, 0))
    cv2.imwrite(name, image)


if __name__ == '__main__':
    # test align loss,
    targetFrame = torch.ones(1, 3, 32, 32)
    GT = torch.ones(1, 3, 32, 32)
    GT += 1
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.

    # referenceFrames = torch.ones(1, 3, 4, 32, 32)
    # referenceMasks = torch.zeros(1, 1, 4, 32, 32)
    # referenceMasks[:, :, 0, 4:12, 4:12] = 1.
    # referenceFrames[:, :, 0, 4:12, 4:12] = 2.
    # referenceMasks[:, :, 1, 4:12, 20:28] = 1.
    # referenceFrames[:, :, 1, 4:12, 20:28] = 2.
    # referenceMasks[:, :, 2, 20:28, 4:12] = 1.
    # referenceFrames[:, :, 2, 20:28, 4:12] = 2.
    # referenceMasks[:, :, 3, 20:28, 20:28] = 1.
    # referenceFrames[:, :, 3, 20:28, 20:28] = 2.
    #
    # aligned_v = referenceMasks
    # aligned_v, referenceFrames = [aligned_v], [referenceFrames]
    #
    # result = AlignLoss()(targetFrame, mask, aligned_v, referenceFrames)
    # print(result)

    c_mask = torch.zeros(1, 1, 32, 32)
    c_mask[:, :, 8:16, 16:24] = 1.
    result1 = HoleVisibleLoss()(targetFrame, mask, GT, c_mask)
    result2 = HoleInvisibleLoss()(targetFrame, mask, GT, c_mask)
    result3 = NonHoleLoss()(targetFrame, mask, GT)
    print('vis: {}, invis: {}, gt: {}'.format(result1, result2, result3))
