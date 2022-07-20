import torch


def flow_prop(feat, flow, mode='forward'):
    """

    Args:
        feat: features to be aligned
        flow: the filled current flow
        mode: `forward` or `backward`, indicates the propagation direction

    Returns: feature after warping

    """
    assert mode in ['forward', 'backward'], 'Invalid mode: {}'.format(mode)
    feat = warp(feat, flow, mode)

    return feat


def warp(feat, flow, mode):
    device = feat.device
    c = feat.shape[1]
    y = flow[:, 0:1, :, :]
    x = flow[:, 1:2, :, :]

    x = x.repeat(1, c, 1, 1)  # [b, c, h, w]
    y = y.repeat(1, c, 1, 1)

    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, y1, x2, y2)

    feat11, o11 = sample_one(feat, x1, y1, w11, mode)
    feat12, o12 = sample_one(feat, x1, y2, w12, mode)
    feat21, o21 = sample_one(feat, x2, y1, w21, mode)
    feat22, o22 = sample_one(feat, x2, y2, w22, mode)

    feat_o = feat11 + feat12 + feat21 + feat22
    o = o11 + o12 + o21 + o22
    feat_o[o > 0] = feat_o[o > 0] / o[o > 0]
    return feat_o


def sample_one(feat, shiftx, shifty, weight, mode):
    device = feat.device
    b, c, h, w = feat.shape
    flat_shiftx = shiftx.view(-1)  # [b * c * h * w]
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, h, requires_grad=False).view(-1, 1).long().repeat(b, c, 1, w).view(-1)
    flat_basey = torch.arange(0, w, requires_grad=False).view(-1, 1).long().repeat(b, c, h, 1).view(-1)
    flat_basex = flat_basex.to(device)
    flat_basey = flat_basey.to(device)
    flat_weight = weight.reshape(-1)
    flat_feat = feat.reshape(-1)

    idxn = torch.arange(0, b, requires_grad=False).view(b, 1, 1, 1).long().repeat(1, c, h, w).view(-1)
    idxc = torch.arange(0, c, requires_grad=False).view(1, c, 1, 1).long().repeat(b, 1, h, w).view(-1)
    idxn = idxn.to(device)
    idxc = idxc.to(device)
    if mode == 'forward':
        idxx = flat_shiftx.long() + flat_basex  # size [-1]
        idxy = flat_shifty.long() + flat_basey  # size [-1]
    else:  # backward propagation
        idxx = -flat_shiftx.long() + flat_basex  # size [-1]
        idxy = -flat_shifty.long() + flat_basey  # size [-1]

    # record the shifted pixels inside the image boundaries
    mask = idxx.ge(0) & idxx.lt(h) & idxy.ge(0) & idxy.lt(w)

    # mask off points out of boundaries
    ids = idxn * c * h * w + idxc * h * w + idxx * w + idxy
    ids_mask = torch.masked_select(ids, mask).clone()

    # put the value into corresponding regions
    feat_warp = torch.zeros([b * c * h * w])
    feat_warp = feat_warp.to(device)
    feat_warp.put_(ids_mask, torch.masked_select(flat_feat * flat_weight, mask), accumulate=True)
    one_warp = torch.zeros([b * c * h * w])
    one_warp = one_warp.to(device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)
    return feat_warp.view(b, c, h, w), one_warp.view(b, c, h, w)


def get_gaussian_weights(x, y, x1, y1, x2, y2):
    sigma = 1
    w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (sigma ** 2))
    w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2) / (sigma ** 2))
    w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2) / (sigma ** 2))
    w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (sigma ** 2))
    return w11, w12, w21, w22
