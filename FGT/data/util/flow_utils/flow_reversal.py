import torch


def flow_reversal(flow):
    """
    flow: shape [b, c, h, w]
    return: backward flow in corresponding to the forward flow
    The formula is borrowed from Quadratic Video Interpolation (4)
    """
    b, c, h, w = flow.shape
    y = flow[:, 0:1, :, :]
    x = flow[:, 1:2, :, :]  # [b, 1, h, w]

    x = x.repeat(1, c, 1, 1)
    y = y.repeat(1, c, 1, 1)

    # get the four points of the square (x1, y1), (x1, y2), (x2, y1), (x2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # calculate the weight maps for each optical flows
    flow11, o11 = sample_one(flow, x1, y1, w11)
    flow12, o12 = sample_one(flow, x1, y2, w12)
    flow21, o21 = sample_one(flow, x2, y1, w21)
    flow22, o22 = sample_one(flow, x2, y2, w22)

    # fuse all the reversed flows based on equation (4)
    flow_o = flow11 + flow12 + flow21 + flow22
    o = o11 + o12 + o21 + o22

    flow_o = -flow_o
    flow_o[o > 0] = flow_o[o > 0] / o[o > 0]

    return flow_o


def get_gaussian_weights(x, y, x1, x2, y1, y2):
    sigma = 1
    w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2) / (sigma ** 2))
    w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2) / (sigma ** 2))
    w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2) / (sigma ** 2))
    w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2) / (sigma ** 2))
    return w11, w12, w21, w22


def sample_one(flow, shiftx, shifty, weight):
    b, c, h, w = flow.shape
    flat_shiftx = shiftx.view(-1)  # [h * w]
    flat_shifty = shifty.view(-1)  # [h * w]
    flat_basex = torch.arange(0, h, requires_grad=False).view(-1, 1).long().repeat(b, c, 1, w).view(-1)  # [h * w]
    flat_basey = torch.arange(0, w, requires_grad=False).view(-1, 1).long().repeat(b, c, h, 1).view(-1)  # [h * w]
    flat_weight = weight.reshape(-1)  # [h * w]
    flat_flow = flow.reshape(-1)

    idxn = torch.arange(0, b, requires_grad=False).view(b, 1, 1, 1).long().repeat(1, c, h, w).view(-1)
    idxc = torch.arange(0, c, requires_grad=False).view(1, c, 1, 1).long().repeat(b, 1, h, w).view(-1)
    idxx = flat_shiftx.long() + flat_basex  # size [-1]
    idxy = flat_shifty.long() + flat_basey  # size [-1]

    # record the shifted pixels inside the image boundaries
    mask = idxx.ge(0) & idxx.lt(h) & idxy.ge(0) & idxy.lt(w)

    # mask off points out of boundaries
    ids = idxn * c * h * w + idxc * h * w + idxx * w + idxy
    ids_mask = torch.masked_select(ids, mask).clone()

    # put the value into corresponding regions
    flow_warp = torch.zeros([b * c * h * w])
    flow_warp.put_(ids_mask, torch.masked_select(flat_flow * flat_weight, mask), accumulate=True)
    one_warp = torch.zeros([b * c * h * w])
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)
    return flow_warp.view(b, c, h, w), one_warp.view(b, c, h, w)
