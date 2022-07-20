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
        self.group_size = group_size  # 这里的group size表示可分的组
        self.wh, self.ww = math.ceil(self.h / self.group_size), math.ceil(self.w / self.group_size)
        self.pad_r = (self.ww - self.w % self.ww) % self.ww
        self.pad_b = (self.wh - self.h % self.wh) % self.wh
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r  # 只在右侧和下侧进行padding，另一侧不padding，实现起来更加容易
        self.window_h, self.window_w = self.new_h // self.group_size, self.new_w // self.group_size  # 这里面的group表示的是窗口大小，而window_size表示的是group大小（与spatial的定义不同）
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


class SMHSA(nn.Module):
    def __init__(self, token_size, d_model, head, p=0.1):
        super(SMHSA, self).__init__()
        self.head = head
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)

    def forward(self, x, t, h=0, w=0):
        bt, n, c = x.shape
        c_h = c // self.head
        query = self.query_embedding(x)  # [bt, n, c']
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        query = query.reshape(bt, n, self.head, c_h).permute(0, 2, 1, 3)
        key = key.reshape(bt, n, self.head, c_h).permute(0, 2, 1, 3)
        value = value.reshape(bt, n, self.head, c_h).permute(0, 2, 1, 3)
        attn, _ = self.attention(query, key, value)  # [b, head, n, c_h]
        attn = attn.permute(0, 2, 1, 3).contiguous().view(bt, n, c)  # [b, n, head, c_h]
        output = self.output_linear(attn)
        return output


class SWMHSA(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2, 3)
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2, 3)
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_globalWindow(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_globalWindow, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=window_size, stride=window_size, padding=0)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_globalWindow2(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_globalWindow2, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=2, stride=2, padding=0)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_globalWindow4(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_globalWindow4, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow4(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow4, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0, groups=d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow4LN(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow4LN, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        global_tokens = self.norm(global_tokens)
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        global_tokens = self.norm(global_tokens)
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow4ConcatLN(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow4ConcatLN, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_globalWindow_maxPool(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_globalWindow_maxPool, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.MaxPool2d(kernel_size=window_size, stride=window_size)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_globalWindow_meanPool(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_globalWindow_meanPool, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.AvgPool2d(kernel_size=window_size, stride=window_size)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_unfoldGlobalWindow4(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_unfoldGlobalWindow4, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0, groups=d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow8ConcatLN(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow8ConcatLN, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=8, stride=8, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow16ConcatLN(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow16ConcatLN, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=16, stride=16, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow4ConcatLN_window1(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow4ConcatLN_window1, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        raise NotImplementedError('Beta version for pixel window version')

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        y = x.view(bt, self.h, self.w, c)
        y = y.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c)  # [b, n2, c]
        query = self.query_embedding(x)  # [bt, n1, c]
        key = self.key_embedding(global_tokens)
        value = self.value_embedding(global_tokens)
        n1, n2 = x.shape[1], global_tokens.shape[1]
        query = query.reshape(bt, n1, self.head,
                              c // self.head).permute(0, 2, 1, 3)
        key = key.reshape(bt, n2, self.head,
                          c // self.head).permute(0, 2, 1, 3)
        value = value.reshape(bt, n2, self.head,
                              c // self.head).permute(0, 2, 1, 3)
        attn, _ = self.attention(query, key, value)  # [b, head, n, c]
        x = attn.transpose(1, 2)
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output


class SWMHSA_depthGlobalWindow2ConcatLN(nn.Module):
    def __init__(self, token_size, window_size, d_model, head, p=0.1):
        super(SWMHSA_depthGlobalWindow2ConcatLN, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract = nn.Conv2d(d_model, d_model, kernel_size=2, stride=2, padding=0, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def inference(self, x, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        x = x.view(bt, h, w, c)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, group_h * group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, group_h * group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, group_h * group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, group_h, group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, group_h * self.window_size, group_w * self.window_size, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output

    def forward(self, x, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, h, w)
        bt, n, c = x.shape
        x = x.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        y = x.permute(0, 3, 1, 2)
        global_tokens = self.global_extract(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        kv = torch.cat((x, global_tokens), dim=2)
        kv = self.norm(kv)
        query = self.query_embedding(x)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(kv)
        value = self.value_embedding(kv)
        query = query.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        key = key.reshape(bt, self.group_h * self.group_w, -1, self.head,
                          c // self.head).permute(0, 1, 3, 2, 4)
        value = value.reshape(bt, self.group_h * self.group_w, -1, self.head,
                              c // self.head).permute(0, 1, 3, 2, 4)
        attn, _ = self.attention(query, key, value)
        x = attn.transpose(2, 3).reshape(bt, self.group_h, self.group_w, self.window_size, self.window_size, c)
        x = x.transpose(2, 3).reshape(bt, self.group_h * self.window_size, self.group_w * self.window_size, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = x[:, :self.h, :self.w, :].contiguous()
        x = x.reshape(bt, n, c)
        output = self.output_linear(x)
        return output
