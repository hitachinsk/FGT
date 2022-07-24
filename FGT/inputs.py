import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="General top layer trainer")
    parser.add_argument("--opt", type=str, default="config/train.yaml", help="Path to optional configuration file")
    parser.add_argument('--model', type=str, default='model',
                        help='Model block name, in the `model` directory')
    parser.add_argument('--name', type=str, default='FGT_train', help='Experiment name')
    parser.add_argument('--outputdir', type=str, default='/myData/ret/experiments', help='Output dir to save results')
    parser.add_argument('--datadir', type=str, default='/myData/', metavar='PATH')
    parser.add_argument('--datasetName_train', type=str, default='train_dataset_frames_diffusedFlows',
                        help='The file name of the train dataset, in `data` directory')
    parser.add_argument('--network', type=str, default='network',
                        help='The network file which defines the training process, in the `network` directory')
    parser.add_argument('--finetune', type=int, default=0, help='Whether to fine tune trained models')
    # parser.add_argument('--checkPoint', type=str, default='', help='checkpoint path for continue training')
    parser.add_argument('--gen_state', type=str, default='', help='Checkpoint of the generator')
    parser.add_argument('--dis_state', type=str, default='', help='Checkpoint of the discriminator')
    parser.add_argument('--opt_state', type=str, default='', help='Checkpoint of the options')
    parser.add_argument('--record_iter', type=int, default=16, help='How many iters to print an item of log')
    parser.add_argument('--flow_checkPoint', type=str, default='flowCheckPoint/flow_model_1290',
                        help='The path for flow model filling')
    parser.add_argument('--dataMode', type=str, default='resize', choices=['resize', 'crop'])

    # data related parameters
    parser.add_argument('--flow2rgb', type=int, default=1, help='Whether to transform flows from raw data to rgb')
    parser.add_argument('--flow_direction', type=str, default='for', choices=['for', 'back', 'bi'],
                        help='Which GT flow should be chosen for guidance')
    parser.add_argument('--num_frames', type=int, default=5, help='How many frames are chosen for frame completion')
    parser.add_argument('--sample', type=str, default='random', choices=['random', 'seq'],
                        help='Choose the sample method for training in each iterations')
    parser.add_argument('--max_val', type=float, default=0.01, help='The maximal value to quantize the optical flows')

    # model related parameters
    parser.add_argument('--res_h', type=int, default=240, help='The height of the frame resolution')
    parser.add_argument('--res_w', type=int, default=432, help='The width of the frame resolution')
    parser.add_argument('--in_channel', type=int, default=4, help='The input channel of the frame branch')
    parser.add_argument('--cnum', type=int, default=64, help='The initial channel number of the frame branch')
    parser.add_argument('--flow_inChannel', type=int, default=2, help='The input channel of the flow branch')
    parser.add_argument('--flow_cnum', type=int, default=64, help='The initial channel dimension of the flow branch')
    parser.add_argument('--dist_cnum', type=int, default=32, help='The initial channel num in the discriminator')
    parser.add_argument('--frame_hidden', type=int, default=512,
                        help='The channel / patch dimension in the frame branch')
    parser.add_argument('--flow_hidden', type=int, default=256, help='The channel / patch dimension in the flow branch')
    parser.add_argument('--PASSMASK', type=int, default=1,
                        help='1 -> concat the mask with the corrupted optical flows to fill the flow')
    parser.add_argument('--numBlocks', type=int, default=8, help='How many transformer blocks do we need to stack')
    parser.add_argument('--kernel_size_w', type=int, default=7, help='The width of the kernel for extracting patches')
    parser.add_argument('--kernel_size_h', type=int, default=7, help='The height of the kernel for extracting patches')
    parser.add_argument('--stride_h', type=int, default=3, help='The height of the stride')
    parser.add_argument('--stride_w', type=int, default=3, help='The width of the stride')
    parser.add_argument('--pad_h', type=int, default=3, help='The height of the padding')
    parser.add_argument('--pad_w', type=int, default=3, help='The width of the padding')
    parser.add_argument('--num_head', type=int, default=4, help='The head number for the multihead attention')
    parser.add_argument('--conv_type', type=str, choices=['vanilla', 'gated', 'partial'], default='vanilla',
                        help='Which kind of conv to use')
    parser.add_argument('--norm', type=str, default='None', choices=['None', 'BN', 'SN', 'IN'],
                        help='The normalization method for the conv blocks')
    parser.add_argument('--use_bias', type=int, default=1, help='If 1, use bias in the convolution blocks')
    parser.add_argument('--ape', type=int, default=1, help='If ape = 1, use absolute positional embedding')
    parser.add_argument('--pos_mode', type=str, default='single', choices=['single', 'dual'],
                        help='If pos_mode = dual, add positional embedding to flow patches')
    parser.add_argument('--mlp_ratio', type=int, default=40, help='The mlp dilation rate for the feed forward layers')
    parser.add_argument('--drop', type=int, default=0, help='The dropout rate, 0 by default')
    parser.add_argument('--init_weights', type=int, default=1, help='If 1, initialize the network, 1 by default')

    # loss related parameters
    parser.add_argument('--L1M', type=float, default=1, help='The weight of L1 loss in the masked area')
    parser.add_argument('--L1V', type=float, default=1, help='The weight of L1 loss in the valid area')
    parser.add_argument('--adv', type=float, default=0.01, help='The weight of adversarial loss')

    # spatial and temporal related parameters
    parser.add_argument('--tw', type=int, default=2, help='The number of temporal group in the temporal transformer')
    parser.add_argument('--sw', type=int, default=8,
                        help='The number of spatial window size in the spatial transformer')
    parser.add_argument('--gd', type=int, default=4, help='Global downsample rate for spatial transformer')

    parser.add_argument('--ref_length', type=int, default=10, help='The sample interval during inference')
    parser.add_argument('--use_valid', action='store_true')

    args = parser.parse_args()
    return args
