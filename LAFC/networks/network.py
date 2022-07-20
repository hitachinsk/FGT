from trainer import Trainer
from importlib import import_module
import math
import torch
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import os
from shutil import copyfile
import glob
from models.utils.flow_losses import smoothness_loss, second_order_loss
from models.utils.fbConsistencyCheck import image_warp
from models.utils.fbConsistencyCheck import ternary_loss2
import torch.nn.functional as F
import cv2
import cvbase
from data.util.flow_utils import region_fill as rf
import imageio
import torch.nn as nn
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.utils.bce_edge_loss import edgeLoss, EdgeAcc


class Network(Trainer):
    def init_model(self):
        self.edgeMeasure = EdgeAcc()
        model_package = import_module('models.{}'.format(self.opt['model']))
        model = model_package.Model(self.opt)
        optimizer = optim.Adam(model.parameters(), lr=float(self.opt['train']['lr']),
                               betas=(float(self.opt['train']['BETA1']), float(float(self.opt['train']['BETA2']))))
        if self.rank <= 0:
            self.logger.info(
                'Optimizer is Adam, BETA1: {}, BETA2: {}'.format(float(self.opt['train']['BETA1']),
                                                                 float(self.opt['train']['BETA2'])))
        step_size = int(math.ceil(self.opt['train']['UPDATE_INTERVAL'] / self.trainSize))
        if self.rank <= 0:
            self.logger.info('Step size for optimizer is {} epoch'.format(step_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.opt['train']['lr_decay'])
        return model, optimizer, scheduler

    def resume_training(self):
        gen_state = torch.load(self.opt['path']['gen_state'],
                                  map_location=lambda storage, loc: storage.cuda(self.opt['device']))
        opt_state = torch.load(self.opt['path']['opt_state'],
                               map_location=lambda storage, loc: storage.cuda(self.opt['device']))
        if self.rank <= 0:
            self.logger.info('Resume state is activated')
            self.logger.info('Resume training from epoch: {}, iter: {}'.format(
                opt_state['epoch'], opt_state['iteration']
            ))
        if self.opt['finetune'] == False:
            start_epoch = opt_state['epoch']
            current_step = opt_state['iteration']
            self.optimizer.load_state_dict(opt_state['optimizer_state_dict'])
            self.scheduler.load_state_dict(opt_state['scheduler_state_dict'])
        else:
            start_epoch = 0
            current_step = 0
        self.model.load_state_dict(gen_state['model_state_dict'])
        if self.rank <= 0:
            self.logger.info('Resume training mode, optimizer, scheduler and model have been uploaded')
        return start_epoch, current_step

    def _trainEpoch(self, epoch):
        for idx, train_data in enumerate(self.trainLoader):
            self.currentStep += 1

            if self.currentStep > self.totalIterations:
                if self.rank <= 0:
                    self.logger.info('Train process has been finished')
                break
            if self.opt['train']['WARMUP'] is not None and self.currentStep <= self.opt['train']['WARMUP'] // self.opt[
                'world_size']:
                target_lr = self.opt['train']['lr'] * self.currentStep / (
                    self.opt['train']['WARMUP'])
                self.adjust_learning_rate(self.optimizer, target_lr)

            flows = train_data['flows']
            diffused_flows = train_data['diffused_flows']
            target_edge = train_data['edges']
            current_frame = train_data['current_frame']
            current_frame = current_frame.to(self.opt['device'])
            shift_frame = train_data['shift_frame']
            shift_frame = shift_frame.to(self.opt['device'])
            masks = train_data['masks']
            flows = flows.to(self.opt['device'])
            masks = masks.to(self.opt['device'])
            diffused_flows = diffused_flows.to(self.opt['device'])
            target_edge = target_edge.to(self.opt['device'])

            b, c, t, h, w = masks.shape
            target_flow = flows[:, :, t // 2]
            target_mask = masks[:, :, t // 2]

            filled_flow = self.model(diffused_flows, masks)

            filled_flow, filled_edge = filled_flow

            combined_flow = target_flow * (1 - target_mask) + filled_flow * target_mask
            combined_edge = target_edge * (1 - target_mask) + filled_edge * target_mask
            edge_loss = (edgeLoss(filled_edge, target_edge) + 5 * edgeLoss(combined_edge, target_edge))

            # loss calculations
            L1Loss_masked = self.maskedLoss(combined_flow * target_mask,
                                            target_flow * target_mask) / torch.mean(target_mask)
            L1Loss_valid = self.validLoss(filled_flow * (1 - target_mask),
                                          target_flow * (1 - target_mask)) / torch.mean(1 - target_mask)

            smoothLoss = smoothness_loss(combined_flow, target_mask)
            smoothLoss2 = second_order_loss(combined_flow, target_mask)
            ternary_loss = self.ternary_loss(combined_flow, target_flow, target_mask, current_frame, shift_frame,
                                             scale_factor=1)

            m_losses = (L1Loss_masked + L1Loss_valid) * self.opt['L1M']
            sm1_loss = smoothLoss * self.opt['sm']
            sm2_loss = smoothLoss2 * self.opt['sm2']
            t_loss = self.opt['ternary'] * ternary_loss
            e_loss = edge_loss * self.opt['edge_loss']

            loss = m_losses + sm1_loss + sm2_loss + t_loss + e_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.opt['gc']:  # gradient clip
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10,
                                         norm_type=2)
            self.optimizer.step()

            if self.opt['use_tb_logger'] and self.rank <= 0 and self.currentStep % 8 == 0:
                print('Mask: {:.03f}, sm: {:.03f}, sm2: {:.03f}, ternary: {:.03f}, edge: {:03f}'.format(
                    m_losses.item(),
                    sm1_loss.item(),
                    sm2_loss.item(),
                    t_loss.item(),
                    e_loss.item()
                ))
                self.tb_logger.add_scalar('{}/recon'.format('train'), m_losses.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('{}/sm'.format('train'), sm1_loss.item(), self.currentStep)
                self.tb_logger.add_scalar('{}/sm2'.format('train'), sm2_loss.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('{}/ternary'.format('train'),
                                          t_loss.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('{}/edge'.format('train'), e_loss.item(),
                                          self.currentStep)

            if self.currentStep % self.opt['logger']['PRINT_FREQ'] == 0 and self.rank <= 0:
                compLog = np.array(combined_flow.detach().permute(0, 2, 3, 1).cpu())
                flowsLog = np.array(target_flow.detach().permute(0, 2, 3, 1).cpu())
                logs = self.calculate_metrics(compLog, flowsLog)
                prec, recall = self.edgeMeasure(filled_edge.detach(), target_edge.detach())
                logs['prec'] = prec
                logs['recall'] = recall
                self._printLog(logs, epoch, loss)

    def ternary_loss(self, comp, flow, mask, current_frame, shift_frame, scale_factor):
        if scale_factor != 1:
            current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode='bilinear')
            shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode='bilinear')
        warped_sc = image_warp(shift_frame, flow)
        noc_mask = torch.exp(-50. * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
        warped_comp_sc = image_warp(shift_frame, comp)
        loss = ternary_loss2(current_frame, warped_comp_sc, noc_mask, mask)
        return loss

    def calculate_metrics(self, results, gts):
        B, H, W, C = results.shape
        psnr_values, ssim_values, L1errors, L2errors = [], [], [], []
        for i in range(B):
            result, gt = results[i], gts[i]
            result_rgb = cvbase.flow2rgb(result)
            gt_rgb = cvbase.flow2rgb(gt)
            psnr_value = psnr(result_rgb, gt_rgb)
            ssim_value = ssim(result_rgb, gt_rgb, multichannel=True)
            residual = result - gt
            L1error = np.mean(np.abs(residual))
            L2error = np.sum(residual ** 2) ** 0.5 / (H * W * C)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            L1errors.append(L1error)
            L2errors.append(L2error)
        psnr_value = np.mean(psnr_values)
        ssim_value = np.mean(ssim_values)
        L1_value = np.mean(L1errors)
        L2_value = np.mean(L2errors)
        return {'l1': L1_value, 'l2': L2_value, 'psnr': psnr_value, 'ssim': ssim_value}

    def _printLog(self, logs, epoch, loss):
        if self.countDown % self.opt['record_iter'] == 0:
            self.total_psnr = 0
            self.total_ssim = 0
            self.total_l1 = 0
            self.total_l2 = 0
            self.total_loss = 0
            self.total_prec = 0
            self.total_recall = 0
            self.countDown = 0
        self.countDown += 1
        message = '[epoch:{:3d}, iter:{:7d}, lr:('.format(epoch, self.currentStep)
        for v in self.get_lr():
            message += '{:.3e}, '.format(v)
        message += ')] '
        self.total_psnr += logs['psnr']
        self.total_ssim += logs['ssim']
        self.total_l1 += logs['l1']
        self.total_l2 += logs['l2']
        self.total_prec += logs['prec'].item()
        self.total_recall += logs['recall'].item()
        self.total_loss += loss.item()
        mean_psnr = self.total_psnr / self.countDown
        mean_ssim = self.total_ssim / self.countDown
        mean_l1 = self.total_l1 / self.countDown
        mean_l2 = self.total_l2 / self.countDown
        mean_prec = self.total_prec / self.countDown
        mean_recall = self.total_recall / self.countDown
        mean_loss = self.total_loss / self.countDown

        message += '{:s}: {:.4e} '.format('mean_loss', mean_loss)
        message += '{:s}: {:} '.format('mean_psnr', mean_psnr)
        message += '{:s}: {:} '.format('mean_ssim', mean_ssim)
        message += '{:s}: {:} '.format('mean_l1', mean_l1)
        message += '{:s}: {:} '.format('mean_l2', mean_l2)
        message += '{:s}: {:} '.format('mean_prec', mean_prec)
        message += '{:s}: {:} '.format('mean_recall', mean_recall)

        if self.opt['use_tb_logger']:
            self.tb_logger.add_scalar('train/mean_psnr', mean_psnr, self.currentStep)
            self.tb_logger.add_scalar('train/mean_ssim', mean_ssim, self.currentStep)
            self.tb_logger.add_scalar('train/mean_l1', mean_l1, self.currentStep)
            self.tb_logger.add_scalar('train/mean_l2', mean_l2, self.currentStep)
            self.tb_logger.add_scalar('train/mean_loss', mean_loss, self.currentStep)
            self.tb_logger.add_scalar('train/mean_prec', mean_prec, self.currentStep)
            self.tb_logger.add_scalar('train/mean_recall', mean_recall, self.currentStep)
        self.logger.info(message)

        if self.currentStep % self.opt['logger']['SAVE_CHECKPOINT_FREQ'] == 0:
            self.save_checkpoint(epoch, 'l1', logs['l1'])

    def save_checkpoint(self, epoch, metric, number):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        gen_state = {
            'model_state_dict': model_state
        }

        opt_state = {
            'epoch': epoch,
            'iteration': self.currentStep,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        gen_name = os.path.join(self.opt['path']['TRAINING_STATE'],
                                 'gen_{}_{}.pth.tar'.format(epoch, self.currentStep))
        opt_name = os.path.join(self.opt['path']['TRAINING_STATE'],
                                'opt_{}_{}.pth.tar'.format(epoch, self.currentStep))
        torch.save(gen_state, gen_name)
        torch.save(opt_state, opt_name)

    def _validate(self, epoch):
        data_path = self.valInfo['data_root']
        mask_path = self.valInfo['mask_root']
        self.model.eval()
        test_list = os.listdir(data_path)
        test_list = test_list[:10]  # only inference 10 videos
        width, height = self.valInfo['flow_width'], self.valInfo['flow_height']
        flow_interval = self.opt['flow_interval']  # The sampling interval for flow completion
        psnr, ssim, l1, l2, prec, recall = {}, {}, {}, {}, {}, {}
        pivot, sequenceLen = 20, self.opt['num_flows']
        for i in range(len(test_list)):
            videoName = test_list[i]
            if self.rank <= 0:
                self.logger.info(f'Video {videoName} is being processed')
            for direction in ['forward_flo', 'backward_flo']:
                flow_dir = os.path.join(data_path, videoName, direction)
                mask_dir = os.path.join(mask_path, videoName)
                flows = self.read_flows(flow_dir, width, height, pivot, sequenceLen, flow_interval)
                masks = self.read_masks(mask_dir, width, height, pivot, sequenceLen, flow_interval)
                if flows == [] or masks == []:
                    if self.rank <= 0:
                        print('Video {} doesn\'t have enough {} flows'.format(videoName, direction))
                    continue
                if self.rank <= 0:
                    self.logger.info('Flows have been read')
                diffused_flows = self.diffusion_filling(flows, masks)
                flows = np.stack(flows, axis=0)
                masks = np.stack(masks, axis=0)
                diffused_flows = np.stack(diffused_flows, axis=0)
                target_flow = flows[self.opt['num_flows'] // 2]
                target_edge = self.load_edge(target_flow)
                target_edge = target_edge[:, :, np.newaxis]
                diffused_flows = torch.from_numpy(np.transpose(diffused_flows, (3, 0, 1, 2))).unsqueeze(
                    0).float()
                masks = torch.from_numpy(np.transpose(masks, (3, 0, 1, 2))).unsqueeze(0).float()
                target_flow = torch.from_numpy(np.transpose(target_flow, (2, 0, 1))).unsqueeze(
                    0).float()
                target_edge = torch.from_numpy(np.transpose(target_edge, (2, 0, 1))).unsqueeze(0).float()
                diffused_flows = diffused_flows.to(self.opt['device'])
                masks = masks.to(self.opt['device'])
                target_flow = target_flow.to(self.opt['device'])
                target_edge = target_edge.to(self.opt['device'])
                target_mask = masks[:, :, sequenceLen // 2]
                with torch.no_grad():
                    filled_flow = self.model(diffused_flows, masks, None)
                filled_flow, filled_edge = filled_flow
                target_diffused_flow = diffused_flows[:, :, sequenceLen // 2]
                combined_flow = target_flow * (1 - target_mask) + filled_flow * target_mask

                # calculate metrics
                psnr_avg, ssim_avg, l1_avg, l2_avg = self.metrics_calc(combined_flow, target_flow)
                prec_avg, recall_avg = self.edgeMeasure(filled_edge, target_edge)
                psnr[videoName] = psnr_avg
                ssim[videoName] = ssim_avg
                l1[videoName] = l1_avg
                l2[videoName] = l2_avg
                prec[videoName] = prec_avg.item()
                recall[videoName] = recall_avg.item()

                # visualize frames and report the phase performance
                if self.rank <= 0:
                    if self.opt['use_tb_logger']:
                        self.tb_logger.add_scalar('test/{}/l1'.format(videoName), l1_avg,
                                                  self.currentStep)
                        self.tb_logger.add_scalar('test/{}/l2'.format(videoName), l2_avg, self.currentStep)
                        self.tb_logger.add_scalar('test/{}/psnr'.format(videoName), psnr_avg, self.currentStep)
                        self.tb_logger.add_scalar('test/{}/ssim'.format(videoName), ssim_avg, self.currentStep)
                        self.tb_logger.add_scalar('test/{}/prec'.format(videoName), prec_avg, self.currentStep)
                        self.tb_logger.add_scalar('test/{}/recall'.format(videoName), recall_avg, self.currentStep)
                        self.vis_flows(combined_flow, target_flow, target_diffused_flow, videoName,
                                       epoch)  # view the difference between diffused flows and the completed flows
                    mean_psnr = np.mean([psnr[k] for k in psnr.keys()])
                    mean_ssim = np.mean([ssim[k] for k in ssim.keys()])
                    mean_l1 = np.mean([l1[k] for k in l1.keys()])
                    mean_l2 = np.mean([l2[k] for k in l2.keys()])
                    mean_prec = np.mean([prec[k] for k in prec.keys()])
                    mean_recall = np.mean([recall[k] for k in recall.keys()])
                    self.logger.info(
                        '[epoch:{:3d}, vid:{}/{}], mean_l1: {:.4e}, mean_l2: {:.4e}, mean_psnr: {:}, mean_ssim: {:}, prec: {:}, recall: {:}'.format(
                            epoch, i, len(test_list), mean_l1, mean_l2, mean_psnr, mean_ssim, mean_prec, mean_recall))

        # give the overall performance
        if self.rank <= 0:
            mean_psnr = np.mean([psnr[k] for k in psnr.keys()])
            mean_ssim = np.mean([ssim[k] for k in ssim.keys()])
            mean_l1 = np.mean([l1[k] for k in l1.keys()])
            mean_l2 = np.mean([l2[k] for k in l2.keys()])
            mean_prec = np.mean([prec[k] for k in prec.keys()])
            mean_recall = np.mean([recall[k] for k in recall.keys()])
            self.logger.info(
                '[epoch:{:3d}], mean_l1: {:.4e} mean_l2: {:.4e} mean_psnr: {:} mean_ssim: {:}, prec: {:}, recall: {:}'.format(
                    epoch, mean_l1, mean_l2, mean_psnr, mean_ssim, mean_prec, mean_recall))
            valid_l1 = mean_l1 + 100
            self.save_checkpoint(epoch, 'l1', valid_l1)

        self.model.train()

    def load_edge(self, flow):
        flow_rgb = cvbase.flow2rgb(flow)
        flow_gray = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
        return canny(flow_gray, sigma=self.opt['datasets']['dataInfo']['edge']['sigma'], mask=None,
                     low_threshold=self.opt['datasets']['dataInfo']['edge']['low_threshold'],
                     high_threshold=self.opt['datasets']['dataInfo']['edge']['high_threshold']).astype(
            np.float)

    def read_flows(self, flow_dir, width, height, pivot, sequenceLen, sample_interval):
        flow_paths = glob.glob(os.path.join(flow_dir, '*.flo'))
        flows = []
        half_seq = sequenceLen // 2
        for i in range(-half_seq, half_seq + 1):
            index = pivot + sample_interval * i
            if index < 0:
                index = 0
            if index >= len(flow_paths):
                index = len(flow_paths) - 1
            flow_path = os.path.join(flow_dir, '{:05d}.flo'.format(index))
            flow = cvbase.read_flow(flow_path)
            pre_height, pre_width = flow.shape[:2]
            flow = cv2.resize(flow, (width, height), cv2.INTER_LINEAR)
            flow[:, :, 0] = flow[:, :, 0] / pre_width * width
            flow[:, :, 1] = flow[:, :, 1] / pre_height * height
            flows.append(flow)
        return flows

    def metrics_calc(self, result, frames):
        psnr_avg, ssim_avg, l1_avg, l2_avg = 0, 0, 0, 0
        result = np.array(result.permute(0, 2, 3, 1).cpu())  # [b, h, w, c]
        gt = np.array(frames.permute(0, 2, 3, 1).cpu())  # [b, h, w, c]
        logs = self.calculate_metrics(result, gt)
        psnr_avg += logs['psnr']
        ssim_avg += logs['ssim']
        l1_avg += logs['l1']
        l2_avg += logs['l2']
        return psnr_avg, ssim_avg, l1_avg, l2_avg

    def read_frames(self, frame_dir, width, height, pivot, sequenceLen):
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        frames = []
        if len(frame_paths) <= 30:
            return frames
        for i in range(pivot, pivot + sequenceLen):
            frame_path = os.path.join(frame_dir, '{:05d}.jpg'.format(i))
            frame = imageio.imread(frame_path)
            frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
            frames.append(frame)
        return frames

    def load_edges(self, frames, width, height):
        edges = []
        for i in range(len(frames)):
            frame = frames[i]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edge = canny(frame_gray, sigma=self.valInfo['sigma'], mask=None,
                         low_threshold=self.valInfo['low_threshold'],
                         high_threshold=self.valInfo['high_threshold']).astype(np.float)  # [h, w, 1]
            edge_t = self.to_tensor(edge, width, height, mode='nearest')
            edges.append(edge_t)
        return edges

    def to_tensor(self, frame, width, height, mode='bilinear'):
        if len(frame.shape) == 2:
            frame = frame[:, :, np.newaxis]
        frame_t = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float()  # [b, c, h, w]
        if width != 0 and height != 0:
            frame_t = F.interpolate(frame_t, size=(height, width), mode=mode)
        return frame_t

    def to_numpy(self, tensor):
        tensor = tensor.cpu()
        tensor = tensor[0]
        array = np.array(tensor.permute(1, 2, 0))
        return array

    def read_masks(self, mask_dir, width, height, pivot, sequenceLen, sample_interval):
        mask_path = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        masks = []
        half_seq = sequenceLen // 2
        for i in range(-half_seq, half_seq + 1):
            index = pivot + i * sample_interval
            if index < 0:
                index = 0
            if index >= len(mask_path):
                index = len(mask_path) - 1
            mask = cv2.imread(mask_path[index], 0)
            mask = mask / 255.
            mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
            mask[mask > 0] = 1
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            assert len(mask.shape) == 3, 'Invalid mask shape: {}'.format(mask.shape)
            masks.append(mask)
        return masks

    def diffusion_filling(self, flows, masks):
        filled_flows = []
        for i in range(len(flows)):
            flow, mask = flows[i], masks[i][:, :, 0]
            flow_filled = np.zeros(flow.shape)
            flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0], mask)
            flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1], mask)
            filled_flows.append(flow_filled)
        return filled_flows

    def vis_flows(self, result, target_flow, diffused_flow, video_name, epoch):
        """
        Vis the filled frames, the GT and the masked frames  with the following format
        |          |          |                        |
        |   Ours   |    GT    |    diffused_flows      |
        |          |          |                        |
        Args:
            result: contains generated flow tensors with shape [1, 2, h, w]
            target_flow: contains GT flow tensors with shape [1, 2, h, w]
            diffused_flow: contains diffused flow tensor with shape [1, 2, h, w]
            video_name: video name
            epoch: epoch

        Returns: No returns, but will save the flows for every flow

        """
        out_root = self.opt['path']['VAL_IMAGES']
        out_dir = os.path.join(out_root, str(epoch), video_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        black_column_pixels = 20
        result = self.to_numpy(result)
        target_flow = self.to_numpy(target_flow)
        diffused_flow = self.to_numpy(diffused_flow)
        result = cvbase.flow2rgb(result)
        target_flow = cvbase.flow2rgb(target_flow)
        diffused_flow = cvbase.flow2rgb(diffused_flow)
        height, width = result.shape[:2]
        canvas = np.zeros((height, width * 3 + black_column_pixels * 2, 3))
        canvas[:, 0:width, :] = result
        canvas[:, width + black_column_pixels: 2 * width + black_column_pixels, :] = target_flow
        canvas[:, 2 * (width + black_column_pixels):, :] = diffused_flow
        imageio.imwrite(os.path.join(out_dir, 'result_compare.png'), canvas)
