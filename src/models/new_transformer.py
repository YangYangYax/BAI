

from torch import nn as nn
import torch.nn.functional as F
import torch
import math
import random
from timm.models.layers import DropPath, trunc_normal_
from .utils import SublayerConnection, BehaviorSpecificPFF, FPM
from .relative_position import RelativePositionBias




class MCL(nn.Module):
    def __init__(self, channels, c2=None, factor=5):
        super(MCL, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)  # 初始化Softmax层，axis=-1表示在最后一个维度上进行Softmax
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 初始化自适应平均池化层，将输入尺寸自适应地池化为(1, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 初始化自适应平均池化层，对高度维度进行平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 初始化自适应平均池化层，对宽度维度进行平均池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 初始化GroupNorm层，对输入的通道维度进行归一化
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 初始化1x1卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 初始化3x3卷积层

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # 重塑输入张量，重新组合通道维度
        x_h = self.pool_h(group_x)  #进行高度维度上的平均池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  #进行宽度维度上的平均池化，并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  #将高度和宽度池化后的结果拼接，经过1x1卷积处理
        x_h, x_w = torch.split(hw, [h, w], dim=2)  #将结果按高度、宽度进行切分
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  #计算得到x1
        x2 = self.conv3x3(group_x)  #对group_x进行3x3卷积操作
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  #通过softmax对x1进行操作
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  #重塑x2的形状
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  #通过softmax对x2进行操作
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  #重塑x1的形状
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  #根据权重计算结果
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  #返回最终计算结果



class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, b_mat=None, rpb=None, W1=None, alpha1=None, W2=None, alpha2=None, mask=None):
        # 1. Calculate Q-K similarity. w. / w.o. multi-behavior dependencies
        if b_mat is not None:
            W1_ = torch.einsum('Bhmn,CBh->Chmn', W1, F.softmax(alpha1, 1))
            att_all = torch.einsum('bhim,Chmn,bhjn->bhijC', query, W1_, key)
            h=W1.size(1)
            scores = att_all.gather(4, b_mat[:,None,:,:,None].repeat(1,h,1,1,1)).squeeze(4) \
                / math.sqrt(query.size(-1)) + rpb
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(query.size(-1)) + rpb

        # 2. dealing with padding and softmax.
        if mask is not None:
            assert len(mask.shape) == 2
            mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            if scores.dtype == torch.float16:
                scores = scores.masked_fill(mask == 0, -65500)
            else:
                scores = scores.masked_fill(mask == 0, -1e30)
        p_attn = self.dropout(nn.functional.softmax(scores, dim=-1))

        # 3. information agregation. w./w.o. multi-behavior dependencies
        if b_mat is not None:
            h=W2.size(1)
            one_hot_b_mat = F.one_hot(b_mat[:,None,:,:], num_classes=alpha2.size(0)).repeat(1,h,1,1,1)
            W2_ = torch.einsum('BhdD,CBh->ChdD', W2, F.softmax(alpha2, 1))
            return torch.einsum('bhij, bhijC, ChdD, bhjd -> bhiD', p_attn, one_hot_b_mat, W2_, value)
            # return torch.matmul(p_attn, value)
        else:
            return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, n_b, battn, brpb,  d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.n_b = n_b
        self.battn = battn
        self.brpb = brpb

        if battn and n_b > 1: # behavior-specific mutual attention
            self.W1 = nn.Parameter(torch.randn(self.n_b, self.h, self.d_k, self.d_k))
            self.alpha1 = nn.Parameter(torch.randn(self.n_b * self.n_b + 1, self.n_b, self.h))
            self.W2 = nn.Parameter(torch.randn(self.n_b, self.h, self.d_k, self.d_k))
            self.alpha2 = nn.Parameter(torch.randn(self.n_b * self.n_b + 1, self.n_b, self.h))
            self.linear_layers = nn.Parameter(torch.randn(3, self.n_b+1, d_model, self.h, self.d_k))
        else:
            self.W1 = None
            self.W2 = None
            self.alpha1, self.alpha2 = None, None
            self.linear_layers = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        self.linear_layers.data.normal_(mean=0.0, std=0.02)

        if self.brpb:
            self.rpb = nn.ModuleList([RelativePositionBias(32,40,self.h) for i in range(self.n_b * self.n_b + 1)])
        self.attention = Attention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, b_seq=None, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        b_mat = ((b_seq[:,:,None]-1)*self.n_b + b_seq[:,None,:]) * (b_seq[:,:,None]*b_seq[:,None,:]!=0)
        # 0. rel pos bias
        if self.brpb:
            rel_pos_bias = torch.stack([layer(seq_len, seq_len) for layer in self.rpb], -1).repeat(batch_size,1,1,1,1)
            rel_pos_bias = rel_pos_bias.gather(4, b_mat[:,None,:,:,None].repeat(1,self.h,1,1,1)).squeeze(4)
        else:
            rel_pos_bias = 0

        if self.battn and self.n_b>1: # behavior-specific mutual attention
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [torch.einsum("bnd, Bdhk, bnB->bhnk", x, self.linear_layers[l], F.one_hot(b_seq,num_classes=self.n_b+1).float())
                             for l, x in zip(range(3), (query, key, value))]
        else:
            # 1) Do all the linear projections in batch from d_model => h x d_k
            query, key, value = [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers[l])
                             for l, x in zip(range(3), (query, key, value))]
            b_mat = None

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, b_mat=b_mat, rpb=rel_pos_bias, W1=self.W1, alpha1=self.alpha1, W2=self.W2, alpha2=self.alpha2, mask=mask)

        # 3) "Concat" using a view.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return x


class ConvBN(torch.nn.Sequential):  # 定义一个名为 ConvBN 的类，继承自 torch.nn.Sequential 类
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):  # 定义类的初始化方法，接受多个参数
        super().__init__()  # 调用父类的初始化方法
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))  # 添加 Conv2d 层到模型
        if with_bn:  # 如果指定使用 batch normalization
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))  # 添加 BatchNorm2d 层到模型
            torch.nn.init.constant_(self.bn.weight, 1)  # 初始化 BatchNorm2d 层的权重为 1
            torch.nn.init.constant_(self.bn.bias, 0)  # 初始化 BatchNorm2d 层的偏置为 0



class EMB(nn.Module):  # 定义一个名为 Block 的类，继承自 nn.Module 类
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):  # 定义类的初始化方法，接受多个参数
        super().__init__()  # 调用父类的初始化方法
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)  # 定义深度可分离卷积层及 Batch Normalization
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)  # 定义一个卷积层用于计算 f1
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)  # 定义一个卷积层用于计算 f2
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)  # 定义卷积层及 Batch Normalization 用于计算 g
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)  # 定义另一个深度可分离卷积层
        self.act = nn.ReLU6()  # 定义激活函数为 ReLU6
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 根据设定的 drop_path 值添加 DropPath 层或者 Identity 层

    def forward(self, x):  # 定义前向传播方法
        input = x  # 保存输入
        x = self.dwconv(x)  # 经过深度可分离卷积层处理
        x1, x2 = self.f1(x), self.f2(x)  # 经过 f1 和 f2 的处理
        x = self.act(x1) * x2  # 激活 f1 的输出后，再与 f2 的输出相乘
        x = self.dwconv2(self.g(x))  # 根据 g 处理 x，并再次经过深度可分离卷积
        x = input + self.drop_path(x)  # 输入与处理后的结果相加，并应用 drop path
        return x  # 返回处理后的结果
    '''
    在这段代码中，首先定义了一个激活函数为 ReLU6 的操作 self.act = nn.ReLU6()，
    然后定义了一个 ConvBN 操作 self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)，
   
    进行了激活函数和乘法操作 x = self.act(x1) * x2，
    最后对 x 进行了深度可分离卷积和 self.g 的操作 x = self.dwconv2(self.g(x))。
    这一系列操作实现了特征的变换、激活以及组合，帮助模型更好地学习和传播特征。
    '''



class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, n_b, battn, bpff, brpb, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param n_b: number of behaviors
        :param battn: use multi-behavior cross attention
        :param bpff: use behavior-specific multi-gated mixture of experts
        :param brpb: use behavior-specific relative position bias
        """
        super().__init__()

        self.start_fc = nn.Linear(in_features=1, out_features=16)
        self.start_fc1 = nn.Linear(50, 16)
        self.attention = MultiHeadedAttention(h=attn_heads, n_b=n_b, battn=battn, brpb=brpb, d_model=hidden, dropout=dropout)
        self.feed_forward = BehaviorSpecificPFF(d_model=hidden, d_ff=feed_forward_hidden, n_b=n_b, bpff=bpff, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.bai =FPM(hidden_size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.projections = nn.Sequential(
            nn.Linear(800, 50)
        )
        self.ema = MCL(channels=50)
        dpr = [x.item() for x in torch.linspace(0, 0.1, 23)]
        random_number = random.randint(0, 20)
        self.blocks = EMB(50, 4, dpr[random_number])



        '''

        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)  # 定义卷积层及 Batch Normalization 用于计算 g
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)  # 定义另一个深度可分离卷积层
        self.act = nn.ReLU6()  # 定义激活函数为 ReLU6
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # 根据设定的 drop_path 值添加 DropPath 层或者 Identity 层
            
        '''



    def forward(self, x, b_seq, mask):

        batch_size = x.shape[0]
        # print(x.shape)#[128, 50, 16]
        x = self.start_fc(x.unsqueeze(-1))
        # print(x.shape)#([128, 50, 16, 16])
        x = self.blocks(x)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 16, -1)
        #([128, 16, 50 ,16])---([128, 16, 800])
        x = self.projections(x).transpose(2, 1)

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, b_seq, mask=mask))

        params=0.7
        x = x * params + self.bai(x) * (1 - params)

        batch_size = x.shape[0]
        x = self.start_fc(x.unsqueeze(-1))
        x = self.ema(x)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 16, -1)
        x = self.projections(x).transpose(2, 1)

        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, b_seq))
        return self.dropout(x)

        return self.dropout(x)