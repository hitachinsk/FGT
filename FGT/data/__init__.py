import logging
import torch
import torch.utils.data
from importlib import import_module


def create_dataloader(phase, dataset, dataset_opt, opt=None, sampler=None):
    logger = logging.getLogger('base')
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * opt['world_size']
        batch_size = dataset_opt['batch_size']
        if sampler is not None:
            logger.info('N_workers: {}, batch_size: {} DDP train dataloader has been established'.format(num_workers,
                                                                                                         batch_size))
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=num_workers, sampler=sampler,
                                               pin_memory=True)
        else:
            logger.info('N_workers: {}, batch_size: {} train dataloader has been established'.format(num_workers,
                                                                                                     batch_size))
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True,
                                               pin_memory=True)

    else:
        logger.info(
            'N_workers: {}, batch_size: {} validate/test dataloader has been established'.format(
                dataset_opt['n_workers'],
                dataset_opt['batch_size']))
        return torch.utils.data.DataLoader(dataset, batch_size=dataset_opt['batch_size'], shuffle=False,
                                           num_workers=dataset_opt['n_workers'],
                                           pin_memory=False)


def create_dataset(dataset_opt, dataInfo, phase, dataset_name):
    if phase == 'train':
        dataset_package = import_module('data.{}'.format(dataset_name))
        dataset = dataset_package.VideoBasedDataset(dataset_opt, dataInfo)

        mode = dataset_opt['mode']
        logger = logging.getLogger('base')
        logger.info(
            '{} train dataset [{:s} - {:s} - {:s}] is created.'.format(dataset_opt['type'].upper(),
                                                                       dataset.__class__.__name__,
                                                                       dataset_opt['name'], mode))
    else:  # validate and test dataset
        return ValueError('No dataset initialized for valdataset')

    return dataset
