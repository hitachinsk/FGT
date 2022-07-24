import math
import parse
import logging
from utils import util
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from data import create_dataset, create_dataloader
from models.utils.loss import *
import yaml
from models.utils.edgeLoss import EdgeLoss
from abc import abstractmethod, ABCMeta


class Trainer(metaclass=ABCMeta):
    def __init__(self, opt, rank):
        self.opt = opt
        self.rank = rank

        # make directory and set logger
        if rank <= 0:
            self.mkdir()
            self.logger, self.tb_logger = self.setLogger()
        self.setSeed()
        self.dataInfo, self.valInfo, self.trainSet, self.trainSize, self.totalIterations, self.totalEpochs, self.trainLoader, self.trainSampler = self.prepareDataset()
        self.model, self.optimizer, self.scheduler = self.init_model()
        self.model = self.model.to(self.opt['device'])
        if opt['path'].get('opt_state', None):
            self.startEpoch, self.currentStep = self.resume_training()
        else:
            self.startEpoch, self.currentStep = 0, 0
        if opt['distributed']:
            self.model = DDP(
                self.model,
                device_ids=[self.opt['local_rank']],
                output_device=self.opt['local_rank'],
                # find_unused_parameters=True
            )
        if self.rank <= 0:
            self.logger.info('Start training from epoch: {}, iter: {}'.format(
                self.startEpoch, self.currentStep))

        self.maskedLoss = nn.L1Loss()
        self.validLoss = nn.L1Loss()
        self.edgeLoss = EdgeLoss(self.opt['device'])
        self.countDown = 0

        # metrics recorder
        self.total_loss = 0
        self.total_psnr = 0
        self.total_ssim = 0
        self.total_l1 = 0
        self.total_l2 = 0

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group['lr']]
        return lr

    def adjust_learning_rate(self, optimizer, target_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr

    def mkdir(self):
        new_name = util.mkdir_and_rename(self.opt['path']['OUTPUT_ROOT'])
        if new_name:
            self.opt['path']['TRAINING_STATE'] = os.path.join(new_name, 'training_state')
            self.opt['path']['LOG'] = os.path.join(new_name, 'log')
            self.opt['path']['VAL_IMAGES'] = os.path.join(new_name, 'val_images')
        if not os.path.exists(self.opt['path']['TRAINING_STATE']):
            os.makedirs(self.opt['path']['TRAINING_STATE'])
        if not os.path.exists(self.opt['path']['LOG']):
            os.makedirs(self.opt['path']['LOG'])
        if not os.path.exists(self.opt['path']['VAL_IMAGES']):
            os.makedirs(self.opt['path']['VAL_IMAGES'])
        # save config file for output
        with open(os.path.join(self.opt['path']['LOG'], 'config.yaml'), 'w') as f:
            yaml.dump(self.opt, f)

    def setLogger(self):
        util.setup_logger('base', self.opt['path']['LOG'], 'train_' + self.opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(parse.toString(self.opt))
        logger.info('OUTPUT DIR IS: {}'.format(self.opt['path']['OUTPUT_ROOT']))
        if self.opt['use_tb_logger']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info('You are using PyTorch {}, Tensorboard will use [tensorboardX)'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(os.path.join(self.opt['path']['OUTPUT_ROOT'], 'log'))
        else:
            tb_logger = None
        return logger, tb_logger

    def setSeed(self):
        seed = self.opt['train']['manual_seed']
        if self.rank <= 0:
            self.logger.info('Random seed: {}'.format(seed))
        util.set_random_seed(seed)
        torch.backends.cudnn.benchmark = True
        if seed == 0:
            torch.backends.cudnn.deterministic = True

    def prepareDataset(self):
        dataInfo = self.opt['datasets']['dataInfo']
        valInfo = self.opt['datasets']['valInfo']
        valInfo['sigma'] = dataInfo['edge']['sigma']
        valInfo['low_threshold'] = dataInfo['edge']['low_threshold']
        valInfo['high_threshold'] = dataInfo['edge']['high_threshold']
        valInfo['norm'] = self.opt['norm']
        if self.rank <= 0:
            self.logger.debug('Val info is: {}'.format(valInfo))
        train_set, train_size, total_iterations, total_epochs = 0, 0, 0, 0
        train_loader, train_sampler = None, None
        for phase, dataset in self.opt['datasets'].items():
            dataset['norm'] = self.opt['norm']
            dataset['dataMode'] = self.opt['dataMode']
            dataset['edge_loss'] = self.opt['edge_loss']
            dataset['ternary'] = self.opt['ternary']
            dataset['num_flows'] = self.opt['num_flows']
            dataset['sample'] = self.opt['sample']
            dataset['use_edges'] = self.opt['use_edges']
            dataset['flow_interval'] = self.opt['flow_interval']
            if phase.lower() == 'train':
                train_set = create_dataset(dataset, dataInfo, phase, self.opt['datasetName_train'])
                train_size = math.ceil(
                    len(train_set) / (dataset['batch_size'] * self.opt['world_size']))  # 计算一个epoch有多少个iterations
                total_iterations = self.opt['train']['MAX_ITERS']
                total_epochs = int(math.ceil(total_iterations / train_size))
                if self.opt['distributed']:
                    train_sampler = DistributedSampler(
                        train_set,
                        num_replicas=self.opt['world_size'],
                        rank=self.opt['global_rank'])
                else:
                    train_sampler = None
                train_loader = create_dataloader(phase, train_set, dataset, self.opt, train_sampler)
                if self.rank <= 0:
                    self.logger.info('Number of training batches: {}, iters: {}'.format(len(train_set),
                                                                                        total_iterations))
                    self.logger.info('Total epoch needed: {} for iters {}'.format(total_epochs, total_iterations))
        assert train_set != 0 and train_size != 0, "Train size cannot be zero"
        assert train_loader is not None, "Cannot find train set, val set can be None"
        return dataInfo, valInfo, train_set, train_size, total_iterations, total_epochs, train_loader, train_sampler

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def resume_training(self):
        pass

    def train(self):
        for epoch in range(self.startEpoch, self.totalEpochs + 1):
            if self.opt['distributed']:
                self.trainSampler.set_epoch(epoch)
            self._trainEpoch(epoch)
            if self.currentStep > self.totalIterations:
                break
            if self.opt['use_valid'] and (epoch + 1) % self.opt['train']['val_freq'] == 0:
                self._validate(epoch)
            self.scheduler.step(epoch)

    @abstractmethod
    def _trainEpoch(self, epoch):
        pass

    @abstractmethod
    def _printLog(self, logs, epoch, loss):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, metric, number):
        pass

    @abstractmethod
    def _validate(self, epoch):
        pass
