import yaml
import os
import os.path as osp


def path_correction(dataInfo, workdir):
    for key in dataInfo.keys():
        if 'path' in key:
            dataInfo[key] = os.path.join(workdir, dataInfo[key])
    return dataInfo


def val_path_correction(valInfo, workdir):
    for key in valInfo.keys():
        if 'root' in key:
            valInfo[key] = os.path.join(workdir, valInfo[key])
    return valInfo


def parse(args_setting, is_train=True):
    opt_path = args_setting['opt']
    print('Current working dir is: {}'.format(os.getcwd()))
    print('There are {} sub directories here'.format(os.listdir(os.getcwd())))
    with open(opt_path, 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    opt['is_train'] = is_train
    opt = {**args_setting, **opt}

    name = opt['name']
    datadir, outputdir = opt['datadir'], opt['outputdir']

    datasets = {}
    for phase, args in opt['datasets'].items():
        # phase is `train`, `val` or `test`
        datasets[phase] = args
        if phase == 'train':
            with open(args['dataInfo_config'], 'r', encoding='utf-8') as f:
                dataInfo = yaml.safe_load(f)
            dataInfo = path_correction(dataInfo, datadir)
            datasets['dataInfo'] = dataInfo
        if phase == 'val':
            with open(args['val_config'], 'r', encoding='utf-8') as f:
                valInfo = yaml.safe_load(f)
            valInfo = val_path_correction(valInfo, datadir)
            datasets['valInfo'] = valInfo
    opt['datasets'] = datasets

    # path
    opt['path'] = {}

    # training settings
    if is_train:
        output_root = osp.join(outputdir, opt['name'], 'experiments')
        opt['path']['OUTPUT_ROOT'] = output_root
        opt['path']['TRAINING_STATE'] = osp.join(output_root, 'training_state')
        opt['path']['LOG'] = osp.join(output_root, 'log')
        opt['path']['VAL_IMAGES'] = osp.join(output_root, 'val_images')
    else:  # for test
        result_root = osp.join(datadir, opt['path']['OUTPUT_ROOT'], 'results', opt['name'])
        opt['path']['RESULT_ROOT'] = osp.join(result_root, 'RESULT_ROOT')
        opt['path']['LOG'] = result_root

    return opt


def toString(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += toString(v, indent_l=1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

