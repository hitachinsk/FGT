from utils.dist import *
from parse import *
from utils.util import find_free_port
import torch.multiprocessing as mp
import torch.distributed
from importlib import import_module

from flow_inputs import args_parser


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

    # dataset file names
    if opt['gen_state'] != '':
        opt['path']['gen_state'] = opt['gen_state']
    if opt['opt_state'] != '':
        opt['path']['opt_state'] = opt['opt_state']

    if args.finetune == 1:
        opt['finetune'] = True
    else:
        opt['finetune'] = False

    print(f'model is: {opt["model"]}')

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
