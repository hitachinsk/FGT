from trainer import Trainer
from importlib import import_module
import math
import torch
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
from metrics import calculate_metrics
import os
import glob
import torch.nn.functional as F
from models.temporal_patch_gan import Discriminator
import cv2
import cvbase
import imageio
from skimage.feature import canny
from models.lafc_single import Model
from data.util.flow_utils import region_fill as rf


class Network(Trainer):
    def init_model(self):
        model_package = import_module('models.{}'.format(self.opt['model']))
        model = model_package.Model(self.opt)
        dist_in = 3
        discriminator = Discriminator(in_channels=dist_in, conv_type=self.opt['conv_type'],
                                      dist_cnum=self.opt['dist_cnum'])
        optimizer = optim.Adam(model.parameters(), lr=float(self.opt['train']['lr']),
                               betas=(float(self.opt['train']['BETA1']), float(float(self.opt['train']['BETA2']))))
        dist_optim = optim.Adam(discriminator.parameters(), lr=float(self.opt['train']['lr']),
                                betas=(float(self.opt['train']['BETA1']), float(float(self.opt['train']['BETA2']))))
        if self.rank <= 0:
            self.logger.info(
                'Optimizer is Adam, BETA1: {}, BETA2: {}'.format(float(self.opt['train']['BETA1']),
                                                                 float(self.opt['train']['BETA2'])))
        step_size = int(math.ceil(self.opt['train']['UPDATE_INTERVAL'] / self.trainSize))
        if self.rank <= 0:
            self.logger.info('Step size for optimizer is {} epoch'.format(step_size))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.opt['train']['lr_decay'])
        dist_scheduler = lr_scheduler.StepLR(dist_optim, step_size=step_size, gamma=self.opt['train']['lr_decay'])
        return model, discriminator, optimizer, dist_optim, scheduler, dist_scheduler

    def init_flow_model(self):
        flow_model = Model(self.opt['flow_config'])
        state = torch.load(self.opt['flow_checkPoint'],
                           map_location=lambda storage, loc: storage.cuda(self.opt['device']))
        flow_model.load_state_dict(state['model_state_dict'])
        flow_model = flow_model.to(self.opt['device'])
        return flow_model

    def resume_training(self):
        gen_state = torch.load(self.opt['path']['gen_state'],
                                  map_location=lambda storage, loc: storage.cuda(self.opt['device']))
        dis_state = torch.load(self.opt['path']['dis_state'],
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
            self.dist_optim.load_state_dict(opt_state['dist_optim_state_dict'])
            self.scheduler.load_state_dict(opt_state['scheduler_state_dict'])
            self.dist_scheduler.load_state_dict(opt_state['dist_scheduler_state_dict'])
        else:
            start_epoch = 0
            current_step = 0
        self.model.load_state_dict(gen_state['model_state_dict'])
        self.dist.load_state_dict(dis_state['dist_state_dict'])
        if self.rank <= 0:
            self.logger.info('Resume training mode, optimizer, scheduler and model have been uploaded')
        return start_epoch, current_step

    def norm_flows(self, flows):
        flattened_flows = flows.flatten(3)
        flow_max = torch.max(flattened_flows, dim=-1, keepdim=True)[0]
        flows = flows / flow_max.unsqueeze(-1)
        return flows

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

            frames = train_data['frames']  # tensor, [b, t, c, h, w]
            masks = train_data['masks']  # tensor, [b, t, c, h, w]
            if self.opt['flow_direction'] == 'for':
                flows = train_data['forward_flo']
            elif self.opt['flow_direction'] == 'back':
                flows = train_data['backward_flo']
            elif self.opt['flow_direction'] == 'bi':
                raise NotImplementedError('Bidirectory flow mode is not implemented')
            else:
                raise ValueError('Unknown flow mode: {}'.format(self.opt['flow_direction']))
            frames = frames.to(self.opt['device'])  # [b, t, c(3), h, w]
            masks = masks.to(self.opt['device'])  # [b, t, 1, h, w]
            flows = flows.to(self.opt['device'])  # [b, t, c(2), h, w]

            b, t, c, h, w = flows.shape
            flows = flows.reshape(b * t, c, h, w)
            compressed_masks = masks.reshape(b * t, 1, h, w)
            with torch.no_grad():
                flows = self.flow_model(flows, compressed_masks)[0]  # filled flows
            flows = flows.reshape(b, t, c, h, w)

            flows = self.norm_flows(flows)

            b, t, c, h, w = frames.shape
            cm, cf = masks.shape[2], flows.shape[2]

            masked_frames = frames * (1 - masks)

            filled_frames = self.model(masked_frames, flows, masks)  # filled_frames shape: [b, t, c, h, w]
            frames = frames.view(b * t, c, h, w)
            masks = masks.view(b * t, cm, h, w)
            comp_img = filled_frames * masks + frames * (1 - masks)

            real_vid_feat = self.dist(frames, t)
            fake_vid_feat = self.dist(comp_img.detach(), t)
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
            dis_loss = (dis_real_loss + dis_fake_loss) / 2
            self.dist_optim.zero_grad()
            dis_loss.backward()
            self.dist_optim.step()

            # calculate generator loss
            gen_vid_feat = self.dist(comp_img, t)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
            gen_loss = gan_loss * self.opt['adv']
            L1Loss_valid = self.validLoss(filled_frames * (1 - masks),
                                          frames * (1 - masks)) / torch.mean(1 - masks)
            L1Loss_masked = self.validLoss(filled_frames * masks,
                                           frames * masks) / torch.mean(masks)
            m_loss_valid = L1Loss_valid * self.opt['L1M']
            m_loss_masked = L1Loss_masked * self.opt['L1V']

            loss = m_loss_valid + m_loss_masked + gen_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.opt['use_tb_logger'] and self.rank <= 0 and self.currentStep % 8 == 0:
                print('Mask: {:.03f}, valid: {:.03f}, dis_fake: {:.03f}, dis_real: {:.03f}, adv: {:.03f}'.format(
                    m_loss_masked.item(),
                    m_loss_valid.item(),
                    dis_fake_loss.item(),
                    dis_real_loss.item(),
                    gen_loss.item()
                ))
            if self.opt['use_tb_logger'] and self.rank <= 0 and self.currentStep % 64 == 0:
                self.tb_logger.add_scalar('{}/recon_mask'.format('train'), m_loss_masked.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('{}/recon_valid'.format('train'), m_loss_valid.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('{}/adv'.format('train'), gen_loss.item(),
                                          self.currentStep)
                self.tb_logger.add_scalar('train/dist', dis_loss.item(), self.currentStep)

            if self.currentStep % self.opt['logger']['PRINT_FREQ'] == 0 and self.rank <= 0:
                c_frames = comp_img.detach().permute(0, 2, 3, 1).cpu()
                f_frames = frames.detach().permute(0, 2, 3, 1).cpu()
                compLog = np.clip(np.array((c_frames + 1) / 2 * 255), 0, 255).astype(np.uint8)
                framesLog = np.clip(np.array((f_frames + 1) / 2 * 255), 0, 255).astype(np.uint8)
                logs = calculate_metrics(compLog, framesLog)
                self._printLog(logs, epoch, loss)

    def _printLog(self, logs, epoch, loss):
        if self.countDown % self.opt['record_iter'] == 0:
            self.total_psnr = 0
            self.total_ssim = 0
            self.total_l1 = 0
            self.total_l2 = 0
            self.total_loss = 0
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
        self.total_loss += loss.item()
        mean_psnr = self.total_psnr / self.countDown
        mean_ssim = self.total_ssim / self.countDown
        mean_l1 = self.total_l1 / self.countDown
        mean_l2 = self.total_l2 / self.countDown
        mean_loss = self.total_loss / self.countDown

        message += '{:s}: {:.4e} '.format('mean_loss', mean_loss)
        message += '{:s}: {:} '.format('mean_psnr', mean_psnr)
        message += '{:s}: {:} '.format('mean_ssim', mean_ssim)
        message += '{:s}: {:} '.format('mean_l1', mean_l1)
        message += '{:s}: {:} '.format('mean_l2', mean_l2)

        if self.opt['use_tb_logger']:
            self.tb_logger.add_scalar('train/mean_psnr', mean_psnr, self.currentStep)
            self.tb_logger.add_scalar('train/mean_ssim', mean_ssim, self.currentStep)
            self.tb_logger.add_scalar('train/mean_l1', mean_l1, self.currentStep)
            self.tb_logger.add_scalar('train/mean_l2', mean_l2, self.currentStep)
            self.tb_logger.add_scalar('train/mean_loss', mean_loss, self.currentStep)
        self.logger.info(message)

        if self.currentStep % self.opt['logger']['SAVE_CHECKPOINT_FREQ'] == 0:
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model,
                                                                       torch.nn.parallel.DistributedDataParallel):
            model_state = self.model.module.state_dict()
            dist_state = self.dist.module.state_dict()
        else:
            model_state = self.model.state_dict()
            dist_state = self.dist.state_dict()
        gen_state = {
            'model_state_dict': model_state
        }
        dis_state = {
            'dist_state_dict': dist_state
        }
        opt_state = {
            'epoch': epoch,
            'iteration': self.currentStep,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dist_optim_state_dict': self.dist_optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dist_scheduler_state_dict': self.dist_scheduler.state_dict()
        }

        gen_name = os.path.join(self.opt['path']['TRAINING_STATE'],
                                 'gen_{}_{}.pth.tar'.format(epoch, self.currentStep))
        dist_name = os.path.join(self.opt['path']['TRAINING_STATE'],
                                'dist_{}_{}.pth.tar'.format(epoch, self.currentStep))
        opt_name = os.path.join(self.opt['path']['TRAINING_STATE'],
                                'opt_{}_{}.pth.tar'.format(epoch, self.currentStep))
        torch.save(gen_state, gen_name)
        torch.save(dis_state, dist_name)
        torch.save(opt_state, opt_name)

    def _validate(self, epoch):
        frame_path = self.valInfo['frame_root']
        mask_path = self.valInfo['mask_root']
        flow_path = self.valInfo['flow_root']
        self.model.eval()
        test_list = os.listdir(flow_path)
        if len(test_list) > 10:
            test_list = test_list[:10]  # only valid 10 videos to save test time
        width, height = self.valInfo['flow_width'], self.valInfo['flow_height']
        psnr, ssim, l1, l2 = {}, {}, {}, {}
        pivot, sequenceLen, ref_length = 20, self.opt['num_frames'], self.opt['ref_length']
        for i in range(len(test_list)):
            videoName = test_list[i]
            if self.rank <= 0:
                self.logger.info('Video {} is been processed'.format(videoName))
            frame_dir = os.path.join(frame_path, videoName)
            mask_dir = os.path.join(mask_path, videoName)
            flow_dir = os.path.join(flow_path, videoName)
            videoLen = len(glob.glob(os.path.join(mask_dir, '*.png')))
            neighbor_ids = [i for i in range(max(0, pivot - sequenceLen // 2), min(videoLen, pivot + sequenceLen // 2))]
            ref_ids = self.get_ref_index(neighbor_ids, videoLen, ref_length)
            ref_ids.extend(neighbor_ids)
            frames = self.read_frames(frame_dir, width, height, ref_ids)
            masks = self.read_masks(mask_dir, width, height, ref_ids)
            flows = self.read_flows(flow_dir, width, height, ref_ids, videoLen - 1)
            if frames == [] or masks == []:
                if self.rank <= 0:
                    print('Video {} doesn\'t have enough frames'.format(videoName))
                continue
            flows = self.diffusion_flows(flows, masks)
            if self.rank <= 0:
                self.logger.info('Frames, masks, and flows have been read')
            frames = np.stack(frames, axis=0)  # [t, h, w, c]
            masks = np.stack(masks, axis=0)
            flows = np.stack(flows, axis=0)
            frames = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).unsqueeze(0).float()
            flows = torch.from_numpy(np.transpose(flows, (0, 3, 1, 2))).unsqueeze(0).float()
            masks = torch.from_numpy(np.transpose(masks, (0, 3, 1, 2))).unsqueeze(0).float()
            frames = frames / 127.5 - 1
            frames = frames.to(self.opt['device'])
            masks = masks.to(self.opt['device'])
            flows = flows.to(self.opt['device'])
            b, t, c, h, w = flows.shape
            flows = flows.reshape(b * t, c, h, w)
            compressed_masks = masks.reshape(b * t, 1, h, w)
            with torch.no_grad():
                flows = self.flow_model(flows, compressed_masks)[0]
            flows = flows.reshape(b, t, c, h, w)
            flows = self.norm_flows(flows)

            b, t, c, h, w = frames.shape
            cm, cf = masks.shape[2], flows.shape[2]
            masked_frames = frames * (1 - masks)
            with torch.no_grad():
                filled_frames = self.model(masked_frames, flows, masks)
            frames = frames.view(b * t, c, h, w)
            masks = masks.view(b * t, cm, h, w)
            comp_img = filled_frames * masks + frames * (1 - masks)  # [t, c, h, w]

            # calculate metrics
            psnr_avg, ssim_avg, l1_avg, l2_avg = self.metrics_calc(comp_img, frames)
            psnr[videoName] = psnr_avg
            ssim[videoName] = ssim_avg
            l1[videoName] = l1_avg
            l2[videoName] = l2_avg

            # visualize frames and report the phase performance
            if self.rank <= 0:
                if self.opt['use_tb_logger']:
                    self.tb_logger.add_scalar('test/{}/l1'.format(videoName), l1_avg,
                                              self.currentStep)
                    self.tb_logger.add_scalar('test/{}/l2'.format(videoName), l2_avg, self.currentStep)
                    self.tb_logger.add_scalar('test/{}/psnr'.format(videoName), psnr_avg, self.currentStep)
                    self.tb_logger.add_scalar('test/{}/ssim'.format(videoName), ssim_avg, self.currentStep)
                masked_frames = masked_frames.view(b * t, c, h, w)
                self.vis_frames(comp_img, masked_frames, frames, videoName,
                                epoch)  # view the difference between diffused flows and the completed flows
                mean_psnr = np.mean([psnr[k] for k in psnr.keys()])
                mean_ssim = np.mean([ssim[k] for k in ssim.keys()])
                mean_l1 = np.mean([l1[k] for k in l1.keys()])
                mean_l2 = np.mean([l2[k] for k in l2.keys()])
                self.logger.info(
                    '[epoch:{:3d}, vid:{}/{}], mean_l1: {:.4e}, mean_l2: {:.4e}, mean_psnr: {:}, mean_ssim: {:}'.format(
                        epoch, i, len(test_list), mean_l1, mean_l2, mean_psnr, mean_ssim))

        # give the overall performance
        if self.rank <= 0:
            mean_psnr = np.mean([psnr[k] for k in psnr.keys()])
            mean_ssim = np.mean([ssim[k] for k in ssim.keys()])
            mean_l1 = np.mean([l1[k] for k in l1.keys()])
            mean_l2 = np.mean([l2[k] for k in l2.keys()])
            self.logger.info(
                '[epoch:{:3d}], mean_l1: {:.4e} mean_l2: {:.4e} mean_psnr: {:} mean_ssim: {:}'.format(
                    epoch, mean_l1, mean_l2, mean_psnr, mean_ssim))
            self.save_checkpoint(epoch)

        self.model.train()

    def get_ref_index(self, neighbor_ids, videoLen, ref_length):
        ref_indices = []
        for i in range(0, videoLen, ref_length):
            if not i in neighbor_ids:
                ref_indices.append(i)
        return ref_indices

    def metrics_calc(self, results, frames):
        psnr_avg, ssim_avg, l1_avg, l2_avg = 0, 0, 0, 0
        results = np.array(results.permute(0, 2, 3, 1).cpu())
        frames = np.array(frames.permute(0, 2, 3, 1).cpu())
        result = np.clip((results + 1) / 2 * 255, 0, 255).astype(np.uint8)
        frames = np.clip((frames + 1) / 2 * 255, 0, 255).astype(np.uint8)
        logs = calculate_metrics(result, frames)
        psnr_avg += logs['psnr']
        ssim_avg += logs['ssim']
        l1_avg += logs['l1']
        l2_avg += logs['l2']
        return psnr_avg, ssim_avg, l1_avg, l2_avg

    def read_frames(self, frame_dir, width, height, ref_indices):
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        frames = []
        if len(frame_paths) <= 30:
            return frames
        for i in ref_indices:
            frame_path = os.path.join(frame_dir, '{:05d}.jpg'.format(i))
            frame = imageio.imread(frame_path)
            frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)
            frames.append(frame)
        return frames

    def diffusion_flows(self, flows, masks):
        assert len(flows) == len(masks), 'Length of flow: {}, length of mask: {}'.format(len(flows), len(masks))
        ret_flows = []
        for i in range(len(flows)):
            flow, mask = flows[i], masks[i]
            flow = self.diffusion_flow(flow, mask)
            ret_flows.append(flow)
        return ret_flows

    def diffusion_flow(self, flow, mask):
        mask = mask[:, :, 0]
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0] * (1 - mask), mask)
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1] * (1 - mask), mask)
        return flow_filled

    def read_flows(self, flow_dir, width, height, ref_ids, frameMaxIndex):
        if self.opt['flow_direction'] == 'for':
            direction = 'forward_flo'
            shift = 0
        elif self.opt['flow_direction'] == 'back':
            direction = 'backward_flo'
            shift = -1
        elif self.opt['flow_direction'] == 'bi':
            raise NotImplementedError('Bidirectional flows processing are not implemented')
        else:
            raise ValueError('Unknown flow direction: {}'.format(self.opt['flow_direction']))
        flows = []
        flow_path = os.path.join(flow_dir, direction)
        for i in ref_ids:
            i += shift
            if i >= frameMaxIndex:
                i = frameMaxIndex - 1
            if i < 0:
                i = 0
            flow_p = os.path.join(flow_path, '{:05d}.flo'.format(i))
            flow = cvbase.read_flow(flow_p)
            pre_height, pre_width = flow.shape[:2]
            flow = cv2.resize(flow, (width, height), cv2.INTER_LINEAR)
            flow[:, :, 0] = flow[:, :, 0] / pre_width * width
            flow[:, :, 1] = flow[:, :, 1] / pre_height * height
            flows.append(flow)
        return flows

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
        array = np.array(tensor.permute(1, 2, 0))
        return array

    def read_masks(self, mask_dir, width, height, ref_indices):
        mask_path = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        masks = []
        if len(mask_path) < 30:
            return masks
        for i in ref_indices:
            mask = cv2.imread(mask_path[i], 0)
            mask = mask / 255.
            mask = cv2.resize(mask, (width, height), cv2.INTER_NEAREST)
            mask[mask > 0] = 1
            mask = mask[:, :, np.newaxis]
            masks.append(mask)
        return masks

    def vis_frames(self, results, masked_frames, frames, video_name, epoch):
        out_root = self.opt['path']['VAL_IMAGES']
        out_dir = os.path.join(out_root, str(epoch), video_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        black_column_pixels = 20
        results, masked_frames, frames = results.cpu(), masked_frames.cpu(), frames.cpu()
        T = results.shape[0]
        for t in range(T):
            result, masked_frame, frame = results[t], masked_frames[t], frames[t]
            result = self.to_numpy(result)
            masked_frame = self.to_numpy(masked_frame)
            frame = self.to_numpy(frame)
            result = np.clip(((result + 1) / 2) * 255, 0, 255)
            frame = np.clip(((frame + 1) / 2) * 255, 0, 255)
            masked_frame = np.clip(((masked_frame + 1) / 2) * 255, 0, 255)  # normalize to [0~255]
            height, width = result.shape[:2]
            canvas = np.zeros((height, width * 3 + black_column_pixels * 2, 3))
            canvas[:, 0:width, :] = result
            canvas[:, width + black_column_pixels: 2 * width + black_column_pixels, :] = frame
            canvas[:, 2 * (width + black_column_pixels):, :] = masked_frame
            imageio.imwrite(os.path.join(out_dir, 'result_compare_{:05d}.png'.format(t)), canvas)
