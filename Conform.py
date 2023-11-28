import time

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


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 2
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv1d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv1d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv1d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv is True:
            self.residual_conv = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x = x + residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool1d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer(inplace=True)

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, L]

        x = self.sample_pooling(x).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1) #sratch token to concatenate with feature

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm1d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer(inplace=True)

    def forward(self, x, L):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, L)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=L * self.up_stride)

class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=4, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, trans2Con=False):

        super(ConvTransBlock, self).__init__()
        expansion = 2
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)


        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion
        self.trans2con = trans2Con
    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, L = x2.shape

        x_st = self.squeeze_block(x2, x_t) #4, 16, 128 and 4, 17, 64 as input 17 for feature with token

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)
        if self.trans2con:
            x_t_r = self.expand_block(x_t, L // self.dw_stride)
            x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class ConTransformer(nn.Module):

    def __init__(self, patch_size = 16, data_length = 128, in_chans=2, num_classes=11, base_channel = 2, channel_ratio=2, num_med_block=0,
                 depth=12, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0.6, attn_drop_rate=0.4, drop_path_rate=0., Device=None, trans2con=False, resconv1=False):

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

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio) # 2
        trans_dw_stride = patch_size  # Framing option : just PatchSize for stride
        embed_dim = base_channel * patch_size
        self.conv_1 = ConvBlock(inplanes=base_channel, outplanes=stage_1_channel, stride=1, res_conv=True) # 2->8 ,128if CR=4
        self.Frame = Frame(channel=base_channel, PatchSize=patch_size, n_patch=self.patch_num, overlap=1, Device=Device) #
        #self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        # 2~4 stage 8->8, 128
        init_stage = 2
        fin_stage = depth // 3 + 1 # 4+1=5 or 3+1=4
        for i in range(init_stage, fin_stage):# stage1->stage1 : if 5 repeat 2,3,4
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, res_conv=resconv1, stride=1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,
                                trans2Con=trans2con
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage 8->16 64
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 5+4 = 9
        for i in range(init_stage, fin_stage): # means 5,6,7,8
            s = 2 if i == init_stage else 1  # decrease the size
            in_channel = stage_1_channel if i == init_stage else stage_2_channel  # stage 1 for the first model
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block,
                                trans2Con=trans2con
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage 16->32 32
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):#means 9 10 11 12
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion,
                                trans2Con=trans2con
                            )
                            )
        self.fin_stage = fin_stage

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
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = x.squeeze(1) # 4, 2, 128

        #x_base = self.act1(self.bn1(self.conv1(x)))

        # 1 stage
        x_t = x_base.transpose(1, 2)  # B, Length, Channel
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.Frame(x_t)
        x_t = torch.cat([cls_tokens, x_t], dim=1)#[:, 0]meas token
        x_t = x_t + self.position
        x_t = self.trans_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

        # conv classification
        #x_p = self.pooling(x).flatten(1)
        #conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return tran_cls

net = ConTransformer(Device='cpu', depth=9, trans2con=True, channel_ratio=2, resconv1=True).eval()
net2 = ConTransformer(Device='cpu', depth=9, trans2con=True, channel_ratio=4, resconv1=False).eval()
net3 = ConTransformer(Device='cpu', depth=12, trans2con=True, channel_ratio=2, resconv1=False).eval()
net4 = ConTransformer(Device='cpu', depth=12, trans2con=True, channel_ratio=4, resconv1=False).eval()
if __name__ == '__main__':
    import numpy as np
    input1 = torch.randn((2, 1, 2, 128))
    def numParams(net):
        num = 0
        for param in net.parameters():
            if param.requires_grad:
                num += int(np.prod(param.size()))
        return num
    start = time.time()
    y = net(input1)[0]
    period = time.time()-start
    print(period)
    # print(numParams(net))
    # print(numParams(net2))
    # print(numParams(net3))
    # print(numParams(net4))

