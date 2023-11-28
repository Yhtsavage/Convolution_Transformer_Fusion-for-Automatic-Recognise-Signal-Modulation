import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

class DB_GLU_block(nn.Module):
    # The DB-GLU scheme for FFN block of the transformer
    def __init__(self,
                 d_model=64,  # The input dimension (d_model=2L, L is the frame length) #set 32
                 dim_feedforward=128,  # The dimension of inner-layer (i.e., the output dimension of W_fc1)
                 dropout=0.,  # The dropout ratio for inner-layer
                 activate='relu',
                 merge_method='add'):  # We present 3 merge methods for two branches ('add' mode perform eq. (6) of the paper)
        super(DB_GLU_block, self).__init__()
        self.merge_method = merge_method
        self.dim_f = dim_feedforward//2
        dim_linear2 = dim_feedforward//2 if merge_method != 'cat' else dim_feedforward
        if merge_method == 'cross_add' and dropout == 0.:
            dropout = 0.1
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout) if merge_method == 'cross_add' else nn.Identity()
        self.linear2 = nn.Linear(dim_linear2, d_model,)
        if activate == 'relu':
            self.Act = nn.ReLU()
        elif activate == 'sigmoid':
            self.Act = nn.Sigmoid()
        else:
            self.Act = nn.GELU()

    def forward(self, x):
        # input format: (Batch, Patch, Dim)
        x_ = self.linear1(x)
        x_1, x_2 = x_[..., :self.dim_f], x_[..., self.dim_f:]
        if self.merge_method == 'add':
            return self.linear2(self.dropout(x_1 * self.Act(x_2) + x_2 * self.Act(x_1)))
        elif self.merge_method == 'cross_add':
            return self.linear2(self.dropout(x_1 * self.Act(x_2)) + self.dropout_2(x_2 * self.Act(x_1)))
        else:
            return self.linear2(self.dropout(torch.cat((x_1 * self.Act(x_2), x_2 * self.Act(x_1)), dim=-1)))


class Frame(nn.Module):  # FEM module
    def __init__(self,
                 channel,
                 PatchSize=4,  # Frame length
                 n_patch=32,  # Frame number
                 overlap=0.5,  # Sliding step length.
                 Device=None,
                 bias=False):
        super(Frame, self).__init__()
        self.Stride=int(PatchSize*overlap)
        self.PatchSize = PatchSize
        self.FrameNum = int(n_patch)
        # self.Embeddinbg = nn.Conv1d(2, PatchSize*2, (PatchSize,), stride=(self.Stride,), bias=bias)
        self.Embedding = nn.Linear(PatchSize*channel, PatchSize*channel)
        self.channel = channel
        self.Device = Device

    def forward(self, x):
        # input_format: (B, 1024, 2)
        # Note that r_I and r_Q are concatenate in the last dimension of x
        Input_Feature = torch.zeros((x.shape[0], self.FrameNum, self.PatchSize * self.channel)).to(self.Device)
        for i in range(self.FrameNum):
            Start = i * self.Stride
            End = Start + self.PatchSize
            Input_Feature[:, i] = torch.cat([x[:, Start:End, c] for c in range(x.shape[2])], dim=-1)
        return self.Embedding(Input_Feature)
        # return self.Embeddinbg(x.permute(0, 2, 1)).permute(0, 2, 1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU(inplace=True), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feedforward = DB_GLU_block(d_model = dim, dim_feedforward= mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.feedforward(self.norm2(x)))
        return x

class LSTMModule(nn.Module):
    def __init__(self, dim, hidden):
        super(LSTMModule, self).__init__()
        self.lstm_module = nn.LSTM(input_size=dim, hidden_size=hidden, batch_first=True)

    def forward(self, x):
        out_Squence, (h, c) = self.lstm_module(x)
        return out_Squence


class RNNTransBlock(nn.Module):
    """
    Basic module for RNNTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, rnn, patch_size, patch_num, device, trans1, trans2):
        super(RNNTransBlock, self).__init__()
        self.RNN = rnn
        self.Frame = Frame(channel=2, PatchSize=patch_size, n_patch=patch_num, Device=device)
        self.degrade = nn.Linear(16, 2)
        self.act = nn.ReLU(inplace=False)
        self.trans_block1 = trans1
        self.trans_block2 = trans2
    def forward(self, x, x_t):
        x_t_1 = self.trans_block1(x_t)
        x_r = self.act(self.RNN(x))
        x_r_m = self.degrade(x_r)
        x_t_r = self.Frame(x_r_m) # B, Patch_num, Embedingchannel
        x_t_r = torch.cat([x_t[:, 0][:, None, :], x_t_r], dim=1) # 4, 16, 128 and 4, 17, 64 as input 17 for feature with token
        x_t = self.trans_block2(x_t_r + x_t + x_t_1)
        return x_r, x_t


class Conformer_AP(nn.Module):

    def __init__(self, patch_size = 16, data_length = 128, in_chans=2, num_classes=11, base_channel = 2, channel_ratio=2, num_med_block=0,
                 depth=12, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0.5, attn_drop_rate=0., drop_path_rate=0., Device=None):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim = base_channel * patch_size  # num_features for consistency with other models
        assert depth % 3 == 0
        self.patch_num = int(data_length / patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.position = nn.Parameter(torch.zeros((1, self.patch_num + 1, embed_dim)))
        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.conv_cls_head = nn.Linear(128, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.degrade = nn.Linear(16, 2)

        # 1 stage
        self.Frame = Frame(channel=base_channel, PatchSize=patch_size, n_patch=self.patch_num, overlap=1, Device=Device)
        self.Block1 = Block(dim=patch_size*base_channel, num_heads=num_heads, drop=drop_rate)
        self.RNN1 = LSTMModule(dim=base_channel, hidden=64)
        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1 #4 means 2-3
        for i in range(init_stage, fin_stage):# stage1->stage1 :
            in_dim = 64 if i==2 else 16
            hidden = 16
            self.add_module('RNN_Trans' + str(i),
                            RNNTransBlock(
                                        rnn = LSTMModule(dim=in_dim, hidden=hidden),
                                        patch_size=patch_size, patch_num=self.patch_num, device=Device,
                                        trans1=Block(dim=patch_size*base_channel, num_heads=4, drop=0.5),
                                        trans2=Block(dim=patch_size*base_channel, num_heads=4, drop=0.5))
                            )
        self.trans_list=[]
        for _ in range(3):
            self.trans_list.append(Block(dim=patch_size*base_channel, num_heads=4, drop=0.5).to(Device))

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        x = x.squeeze()
        x_cplx = x[:, 0] + 1j * x[:, 1]
        # input feature

        x_angle = torch.angle(x_cplx)
        x_abs = torch.abs(x_cplx)
        x_ap = torch.stack((x_angle, x_abs), dim=1)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #x_base = x.squeeze(1) # 4, 2, 128
        x_base = x_ap.transpose(1, 2)
        #x_base = self.act1(self.bn1(self.conv1(x)))

        # 1 stage  # B, Length, Channel
        x_r = self.RNN1(x_base) # B, L, C
        x_t = self.Frame(x_base)
        x_t = torch.cat([cls_tokens, x_t], dim=1)#[:, 0]meas token
        x_t = x_t + self.position
        x_t = self.Block1(x_t)

        # 2 ~ final
        for i in range(2, 4):
            x_r, x_t = eval('self.RNN_Trans' + str(i))(x_r, x_t)
        x_r = self.act(self.degrade(x_r))
        x_t = x_t + torch.cat([x_t[:, 0][:, None, :], self.Frame(x_r)], dim=1)
        # conv classification
        #x_p = self.pooling(x).flatten(1)
        #conv_cls = self.conv_cls_head(x_p)
        #only trans
        for i in range(len(self.trans_list)):
            x_t= self.trans_list[i](x_t)
        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return tran_cls

#net = Conformer_AP(Device='cpu')
if __name__ == '__main__':
    import numpy as np
    net = Conformer_AP(Device='cpu')
    input1 = torch.randn((4, 1, 2, 128))
    def numParams(net):
        num = 0
        for param in net.parameters():
            if param.requires_grad:
                num += int(np.prod(param.size()))
        return num
    y = net(input1)
    print(numParams(net))

