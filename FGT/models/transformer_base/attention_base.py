import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class TMHSA(nn.Module):
    def __init__(self, token_size, group_size, d_model, head, p=0.1):
        super(TMHSA, self).__init__()
        self.h, self.w = token_size
        self.group_size = group_size
        self.wh, self.ww = math.ceil(self.h / self.group_size), math.ceil(self.w / self.group_size)
        self.pad_r = (self.ww - self.w % self.ww) % self.ww
        self.pad_b = (self.wh - self.h % self.wh) % self.wh
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.window_h, self.window_w = self.new_h // self.group_size, self.new_w // self.group_size
        self.d_model = d_model
        self.p = p
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def inference(self, x, t, h, w):
        # calculate the attention related parameters
        wh, ww = math.ceil(h / self.group_size), math.ceil(w / self.group_size)
        pad_r = (ww - w % ww) % ww
        pad_b = (wh - h % wh) % wh
        new_h, new_w = h + pad_b, w + pad_r
        window_h, window_w = new_h // self.group_size, new_w // self.group_size
        bt, n, c = x.shape
        b = bt // t
        c_h = c // self.head
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x,
                      (0, 0, 0, pad_r, 0, pad_b))  # channel, channel, left, right, top, bottom -> [bt, new_h, new_w, c]
        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        query = query.view(b, t, self.group_size, window_h, self.group_size, window_w, self.head, c_h)
        query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        key = key.view(b, t, self.group_size, window_h, self.group_size, window_w, self.head, c_h)
        key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        value = value.view(b, t, self.group_size, window_h, self.group_size, window_w, self.head, c_h)
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        att, _ = self.attention(query, key, value)
        att = att.view(b, self.group_size, self.group_size, self.head, t, window_h, window_w, c_h)
        att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, new_h, new_w, c)
        if pad_b > 0 or pad_r > 0:
            att = att[:, :h, :w, :]
        att = att.reshape(bt, n, c)
        output = self.output_linear(att)
        return output

    def forward(self, x, t, h=0, w=0):
        bt, n, c = x.shape
        if h == 0 and w == 0:
            assert n == self.h * self.w, 'Wrong input shape: {} with token: h->{}, w->{}'.format(x.shape, self.h,
                                                                                                 self.w)
        else:
            assert n == h * w, 'Wrong input shape: {} with token: h->{}, w->{}'.format(x.shape, h, w)
            return self.inference(x, t, h, w)
        b = bt // t
        c_h = c // self.head
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (
            0, 0, 0, self.pad_r, 0, self.pad_b))  # channel, channel, left, right, top, bottom -> [bt, new_h, new_w, c]
        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        query = query.view(b, t, self.group_size, self.window_h, self.group_size, self.window_w, self.head, c_h)
        query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        key = key.view(b, t, self.group_size, self.window_h, self.group_size, self.window_w, self.head, c_h)
        key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        value = value.view(b, t, self.group_size, self.window_h, self.group_size, self.window_w, self.head, c_h)
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)
        att, _ = self.attention(query, key, value)
        att = att.view(b, self.group_size, self.group_size, self.head, t, self.window_h, self.window_w, c_h)
        att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, self.new_h, self.new_w, c)
        if self.pad_b > 0 or self.pad_r > 0:
            att = att[:, :self.h, :self.w, :]
        att = att.reshape(bt, n, c)
        output = self.output_linear(att)
        return output
