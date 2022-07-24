import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="General top layer trainer")
    parser.add_argument("--opt", type=str, default="config/train.yaml", help="Path to optional configuration file")
    parser.add_argument('--model', type=str, default='2d_net_appendEdges',
                        help='Model block name, in the `model` directory')
    parser.add_argument('--name', type=str, default='flowFusion', help='Experiment name')
    parser.add_argument('--outputdir', type=str, default='/myData/ret/experiments', help='Output dir to save results')
    parser.add_argument('--datadir', type=str, default='/myData/', metavar='PATH')
    parser.add_argument('--datasetName_train', type=str, default='train_dataset_single_edge',
                        help='The file name of the train dataset, in `data` directory')
    parser.add_argument('--dataMode', type=str, default='resize', choices=['resize', 'crop'])
    parser.add_argument('--network', type=str, default='flow_single_appendEdges',
                        help='The network file which defines the training process, in the `network` directory')
    parser.add_argument('--PASSMASK', type=int, default=1,
                        help='1 -> concat the mask with the corrupted optical flows to fill the flow')
    parser.add_argument('--L1M', type=float, default=1, help='The weight of L1 loss in masked area')
    parser.add_argument('--sm', type=float, default=1, help='The loss weight of smooth loss')
    parser.add_argument('--sm2', type=float, default=1, help='The loss weight of second order smooth loss')
    # model related parameters
    parser.add_argument('--use_bias', type=int, default=1, help='If 1, use bias in the convolution blocks')
    parser.add_argument('--norm', type=int, default=0, help='If 1, normalize the weights of layers (SN by deault)')
    parser.add_argument('--init_weights', type=int, default=1,
                        help='If 1, initialize the network with kaiming parameter set.')
    parser.add_argument('--cnum', type=int, default=48, help='Initial channel of encoder')
    parser.add_argument('--finetune', type=int, default=0, help='Whether to fine tune trained models')
    # parser.add_argument('--checkPoint', type=str, default='', help='checkpoint path for continue training')
    parser.add_argument('--gen_state', type=str, default='', help='checkpoint of the gen state for continuous training')
    parser.add_argument('--opt_state', type=str, default='', help='checkpoint of the opt state for continuous training')
    parser.add_argument('--resBlocks', type=int, default=1, help='The number of resblocks in the modulation subnetwork')
    parser.add_argument('--edge_residuals', type=int, default=4, help='The number of resblocks in edge branch')
    parser.add_argument('--conv_type', type=str, choices=['vanilla', 'gated', 'partial'], default='vanilla',
                        help='Which kind of conv to use')
    parser.add_argument('--edge_loss', type=float, default=1, help='Loss weight for edge loss')
    parser.add_argument('--in_channel', type=int, default=3, help='The input channel of the defined network')
    parser.add_argument('--record_iter', type=int, default=16, help='How many iters to print an item of log')
    parser.add_argument('--num_flows', type=int, default=1)
    parser.add_argument('--sample', type=str, default='seq', choices=['random', 'seq'])
    parser.add_argument('--use_edges', type=int, default=0)
    parser.add_argument('--flow_interval', type=int, default=1)
    parser.add_argument('--use_residual', type=int, default=1, help='Whether to use residual for the P3D network')
    parser.add_argument('--gc', type=int, default=0, help='Use gradient clip to stabilize training')
    parser.add_argument('--rescale', type=int, default=1,
                        help='Whether to rescale the results to the original flow range for calculating losses?')
    parser.add_argument('--ternary', type=float, default=0.01)
    parser.add_argument('--use_valid', action='store_true')
    args = parser.parse_args()
    return args
