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


class SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFlow(nn.Module):
    def __init__(self, token_size, window_size, kernel_size, d_model, flow_dModel, head, p=0.1):
        super(SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFlow, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.flow_dModel = flow_dModel
        in_channels = d_model + flow_dModel
        self.query_embedding = nn.Linear(in_channels, d_model)
        self.key_embedding = nn.Linear(in_channels, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract_v = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, stride=kernel_size, padding=0,
                                          groups=d_model)
        self.global_extract_k = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size,
                                          padding=0,
                                          groups=in_channels)
        self.q_norm = nn.LayerNorm(d_model + flow_dModel)
        self.k_norm = nn.LayerNorm(d_model + flow_dModel)
        self.v_norm = nn.LayerNorm(d_model)
        self.reweightFlow = nn.Sequential(
            nn.Linear(in_channels, flow_dModel),
            nn.Sigmoid()
        )

    def inference(self, x, f, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        cf = f.shape[2]
        x = x.view(bt, h, w, c)
        f = f.view(bt, h, w, cf)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
            f = F.pad(f, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        xf = torch.cat((x, f), dim=-1)
        flow_weights = self.reweightFlow(xf)
        f = f * flow_weights
        qk = torch.cat((x, f), dim=-1)  # [b, h, w, c]
        qk_c = qk.shape[-1]
        # generate q
        q = qk.reshape(bt, group_h, self.window_size, group_w, self.window_size, qk_c).transpose(2, 3)
        q = q.reshape(bt, group_h * group_w, self.window_size * self.window_size, qk_c)
        # generate k
        ky = qk.permute(0, 3, 1, 2)  # [b, c, h, w]
        k_global = self.global_extract_k(ky)
        k_global = k_global.permute(0, 2, 3, 1).reshape(bt, -1, qk_c).unsqueeze(1).repeat(1, group_h * group_w, 1, 1)
        k = torch.cat((q, k_global), dim=2)
        # norm q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # generate v
        global_tokens = self.global_extract_v(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        v = torch.cat((x, global_tokens), dim=2)
        v = self.v_norm(v)
        query = self.query_embedding(q)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(k)
        value = self.value_embedding(v)
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

    def forward(self, x, f, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, f, h, w)
        bt, n, c = x.shape
        cf = f.shape[2]
        x = x.view(bt, self.h, self.w, c)
        f = f.view(bt, self.h, self.w, cf)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
            f = F.pad(f, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))  # [bt, cf, h, w]
        y = x.permute(0, 3, 1, 2)
        xf = torch.cat((x, f), dim=-1)
        weights = self.reweightFlow(xf)
        f = f * weights
        qk = torch.cat((x, f), dim=-1)  # [b, h, w, c]
        qk_c = qk.shape[-1]
        # generate q
        q = qk.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, qk_c).transpose(2, 3)
        q = q.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, qk_c)
        # generate k
        ky = qk.permute(0, 3, 1, 2)  # [b, c, h, w]
        k_global = self.global_extract_k(ky)  # [b, qk_c, h, w]
        k_global = k_global.permute(0, 2, 3, 1).reshape(bt, -1, qk_c).unsqueeze(1).repeat(1,
                                                                                          self.group_h * self.group_w,
                                                                                          1, 1)
        k = torch.cat((q, k_global), dim=2)
        # norm q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # generate v
        global_tokens = self.global_extract_v(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        v = torch.cat((x, global_tokens), dim=2)
        v = self.v_norm(v)
        query = self.query_embedding(q)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(k)
        value = self.value_embedding(v)
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


class SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFF(nn.Module):
    def __init__(self, token_size, window_size, kernel_size, d_model, flow_dModel, head, p=0.1):
        super(SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFF, self).__init__()
        self.h, self.w = token_size
        self.head = head
        self.window_size = window_size  # 这里的window size指的是小窗口的大小
        self.d_model = d_model
        self.flow_dModel = flow_dModel
        in_channels = d_model + flow_dModel
        self.query_embedding = nn.Linear(in_channels, d_model)
        self.key_embedding = nn.Linear(in_channels, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p)
        self.pad_l = self.pad_t = 0
        self.pad_r = (self.window_size - self.w % self.window_size) % self.window_size
        self.pad_b = (self.window_size - self.h % self.window_size) % self.window_size
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.group_h, self.group_w = self.new_h // self.window_size, self.new_w // self.window_size
        self.global_extract_v = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, stride=kernel_size, padding=0,
                                          groups=d_model)
        self.global_extract_k = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size,
                                          padding=0,
                                          groups=in_channels)
        self.q_norm = nn.LayerNorm(d_model + flow_dModel)
        self.k_norm = nn.LayerNorm(d_model + flow_dModel)
        self.v_norm = nn.LayerNorm(d_model)
        self.reweightFlow = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def inference(self, x, f, h, w):
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        new_h, new_w = h + pad_b, w + pad_r
        group_h, group_w = new_h // self.window_size, new_w // self.window_size
        bt, n, c = x.shape
        cf = f.shape[2]
        x = x.view(bt, h, w, c)
        f = f.view(bt, h, w, cf)
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
            f = F.pad(f, (0, 0, self.pad_l, pad_r, self.pad_t, pad_b))
        y = x.permute(0, 3, 1, 2)
        qk = torch.cat((x, f), dim=-1)  # [b, h, w, c]
        weights = self.reweightFlow(qk)
        qk = qk * weights
        qk_c = qk.shape[-1]
        # generate q
        q = qk.reshape(bt, group_h, self.window_size, group_w, self.window_size, qk_c).transpose(2, 3)
        q = q.reshape(bt, group_h * group_w, self.window_size * self.window_size, qk_c)
        # generate k
        ky = qk.permute(0, 3, 1, 2)  # [b, c, h, w]
        k_global = self.global_extract_k(ky)
        k_global = k_global.permute(0, 2, 3, 1).reshape(bt, -1, qk_c).unsqueeze(1).repeat(1, group_h * group_w, 1, 1)
        k = torch.cat((q, k_global), dim=2)
        # norm q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # generate v
        global_tokens = self.global_extract_v(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 group_h * group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, group_h, self.window_size, group_w, self.window_size, c).transpose(2,
                                                                                             3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, group_h * group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        v = torch.cat((x, global_tokens), dim=2)
        v = self.v_norm(v)
        query = self.query_embedding(q)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(k)
        value = self.value_embedding(v)
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

    def forward(self, x, f, t, h=0, w=0):
        if h != 0 or w != 0:
            return self.inference(x, f, h, w)
        bt, n, c = x.shape
        cf = f.shape[2]
        x = x.view(bt, self.h, self.w, c)
        f = f.view(bt, self.h, self.w, cf)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
            f = F.pad(f, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))  # [bt, cf, h, w]
        y = x.permute(0, 3, 1, 2)
        qk = torch.cat((x, f), dim=-1)  # [b, h, w, c]
        weights = self.reweightFlow(qk)
        qk = qk * weights
        qk_c = qk.shape[-1]
        # generate q
        q = qk.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, qk_c).transpose(2, 3)
        q = q.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, qk_c)
        # generate k
        ky = qk.permute(0, 3, 1, 2)  # [b, c, h, w]
        k_global = self.global_extract_k(ky)  # [b, qk_c, h, w]
        k_global = k_global.permute(0, 2, 3, 1).reshape(bt, -1, qk_c).unsqueeze(1).repeat(1,
                                                                                          self.group_h * self.group_w,
                                                                                          1, 1)
        k = torch.cat((q, k_global), dim=2)
        # norm q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # generate v
        global_tokens = self.global_extract_v(y)  # [bt, c, h', w']
        global_tokens = global_tokens.permute(0, 2, 3, 1).reshape(bt, -1, c).unsqueeze(1).repeat(1,
                                                                                                 self.group_h * self.group_w,
                                                                                                 1,
                                                                                                 1)  # [bt, gh * gw, h'*w', c]
        x = x.reshape(bt, self.group_h, self.window_size, self.group_w, self.window_size, c).transpose(2,
                                                                                                       3)  # [bt, gh, gw, ws, ws, c]
        x = x.reshape(bt, self.group_h * self.group_w, self.window_size * self.window_size, c)  # [bt, gh * gw, ws^2, c]
        v = torch.cat((x, global_tokens), dim=2)
        v = self.v_norm(v)
        query = self.query_embedding(q)  # [bt, self.group_h, self.group_w, self.window_size, self.window_size, c]
        key = self.key_embedding(k)
        value = self.value_embedding(v)
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
