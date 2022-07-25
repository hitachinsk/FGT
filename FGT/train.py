from utils.dist import *
from parse import *
from utils.util import find_free_port
import torch.multiprocessing as mp
import torch.distributed
from importlib import import_module
import os
import glob
from inputs import args_parser


def main_worker(rank, opt):
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = rank
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch')
        print('using GPU {}-{} for training'.format(
            int(opt['global_rank']), int(opt['local_rank'])))

    if torch.cuda.is_available():
        opt['device'] = torch.device("cuda:{}".format(opt['local_rank']))
    else:
        opt['device'] = 'cpu'

    pkg = import_module('networks.{}'.format(opt['network']))
    trainer = pkg.Network(opt, rank)
    trainer.train()


def main(args_obj):
    opt = parse(args_obj)
    opt['world_size'] = get_world_size()
    free_port = find_free_port()
    master_ip = get_master_ip()
    opt['init_method'] = "tcp://{}:{}".format(master_ip, free_port)
    opt['distributed'] = True if opt['world_size'] > 1 else False
    print(f'World size is: {opt["world_size"]}, and init_method is: {opt["init_method"]}')
    print('Import network module: ', opt['network'])

    checkpoint, config = glob.glob(os.path.join(opt['flow_checkPoint'], '*.tar'))[0], \
                         glob.glob(os.path.join(opt['flow_checkPoint'], '*.yaml'))[0]
    with open(config, 'r') as f:
        configs = yaml.load(f)
    opt['flow_config'] = configs
    opt['flow_checkPoint'] = checkpoint

    if args.finetune == 1:
        opt['finetune'] = True
    else:
        opt['finetune'] = False
    if opt['gen_state'] != '':
        opt['path']['gen_state'] = opt['gen_state']
    if opt['dis_state'] != '':
        opt['path']['dis_state'] = opt['dis_state']
    if opt['opt_state'] != '':
        opt['path']['opt_state'] = opt['opt_state']

    opt['input_resolution'] = (opt['res_h'], opt['res_w'])
    opt['kernel_size'] = (opt['kernel_size_h'], opt['kernel_size_w'])
    opt['stride'] = (opt['stride_h'], opt['stride_w'])
    opt['padding'] = (opt['pad_h'], opt['pad_w'])

    print('model is: {}'.format(opt['model']))

    if get_master_ip() == "127.0.0.1":
        # localhost
        mp.spawn(main_worker, nprocs=opt['world_size'], args=(opt,))
    else:
        # multiple processes should be launched by openmpi
        opt['local_rank'] = get_local_rank()
        opt['global_rank'] = get_global_rank()
        main_worker(-1, opt)


if __name__ == '__main__':
    args = args_parser()
    args_obj = vars(args)
    main(args_obj)
