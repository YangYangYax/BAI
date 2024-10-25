import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# class RelativePositionBias(nn.Module):
#     def __init__(self, num_buckets=32, max_distance=128, n_heads=2):
#         super(RelativePositionBias, self).__init__()
#         # 初始化相对位置偏置模块
#         self.num_buckets = num_buckets  # 定义桶的数量
#         self.max_distance = max_distance  # 定义最大距离
#         self.relative_attention_bias = nn.Embedding(self.num_buckets, n_heads)  # 创建一个Embedding层，用于存储相对位置偏置
#
#     @staticmethod
#     def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
#         # 将相对位置映射到相应的桶中
#         ret = 0
#         n = -relative_position  # 计算相对位置的负值
#         num_buckets //= 2
#         ret += (n < 0).to(torch.long) * num_buckets  # 将小于0的值映射到前半部分的桶中
#         n = torch.abs(n)  # 计算绝对值
#
#         max_exact = num_buckets // 2  # 定义精确位置的桶数
#         is_small = n < max_exact  # 是否是小于max_exact的位置
#
#         val_if_large = max_exact + (
#             torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
#         ).long()  # 根据相对位置的大小，映射到相应的桶
#         val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
#
#         ret += torch.where(is_small, n, val_if_large)  # 根据is_small选择相应的桶
#         return ret
#
#     def forward(self, qlen, klen):
#         # 计算相对位置偏置
#         device = self.relative_attention_bias.weight.device
#         q_pos = torch.arange(qlen, dtype=torch.long, device=device)  # 生成查询位置
#         k_pos = torch.arange(klen, dtype=torch.long, device=device)  # 生成键位置
#         relative_position = k_pos[None, :] - q_pos[:, None]  # 计算相对位置
#         rp_bucket = self._relative_position_bucket(
#             relative_position,  # 相对位置
#             num_buckets=self.num_buckets,  # 桶的数量
#             max_distance=self.max_distance,  # 最大距离
#         )
#         rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
#         values = self.relative_attention_bias(rp_bucket)  # 从Embedding层中获取相对位置偏置值
#         values = values.permute([2, 0, 1]).unsqueeze(0)  # 调整形状以适应注意力机制的输入要求
#         return values
#
#
#





'''
尝试消融代码
'''
class RelativePositionBias(nn.Module):
    # def __init__(self, n_heads=2):
    #     super(RelativePositionBias, self).__init__()
    #     self.relative_attention_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

    def __init__(self, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        # 初始化相对位置偏置模块
        self.num_buckets = num_buckets  # 定义桶的数量
        self.max_distance = max_distance  # 定义最大距离
        self.relative_attention_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        # self.relative_attention_bias = nn.Embedding(self.num_buckets, n_heads)  # 创建一个Embedding层，用于存储相对位置偏置

    def forward(self, qlen, klen):
        device = self.relative_attention_bias.device
        q_pos = torch.arange(qlen, dtype=torch.long, device=device).unsqueeze(-1)
        k_pos = torch.arange(klen, dtype=torch.long, device=device).unsqueeze(-1)
        relative_position = k_pos - q_pos.transpose(0, 1)
        values = self.relative_attention_bias.repeat(1, 1, qlen, klen)
        # print("---------------------------RelativePositionBias-------------------------------------")
        return values

