
from torch import nn as nn

class PositionalEmbedding(nn.Module):
    # PositionalEmbedding 类，用于生成位置嵌入

    def __init__(self, max_len, d_model):
        super().__init__()
        # 初始化父类 nn.Module
        self.pe = nn.Embedding(max_len, d_model)
        # 创建一个嵌入层，max_len 表示位置的最大长度，d_model 表示嵌入向量的维度
        self.apply(self._init_weights)
        # 应用 _init_weights 函数初始化权重

    def forward(self, x):
        batch_size = x.size(0)
        # 获取输入 x 的批量大小
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # 返回位置嵌入权重，先增加一个批量维度，然后复制扩展到整个批量大小

    def _init_weights(self, module):
        """Initialize the weights."""
        # 初始化权重的函数
        if isinstance(module, nn.Embedding):
            # 如果模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 使用均值为 0，标准差为 0.02 的正态分布初始化权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果有填充索引，则将对应的权重设置为 0

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """
    # BERTEmbedding 类，用于生成 BERT 的嵌入表示，包括词嵌入和位置嵌入

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # 初始化父类 nn.Module
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        # 创建词嵌入层，vocab_size 是词汇表大小，embed_size 是嵌入向量维度，padding_idx 是填充索引
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # 创建位置嵌入对象，max_len 是序列的最大长度
        self.dropout = nn.Dropout(p=dropout)
        # 创建 dropout 层，dropout 参数为丢弃率
        self.embed_size = embed_size
        # 保存嵌入向量的维度
        self.apply(self._init_weights)
        # 应用 _init_weights 函数初始化权重

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        # 对输入序列进行词嵌入和位置嵌入，并相加
        return self.dropout(x)
        # 应用 dropout 并返回最终嵌入结果

    def _init_weights(self, module):
        """Initialize the weights."""
        # 初始化权重的函数
        if isinstance(module, nn.Embedding):
            # 如果模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=0.02)
            # 使用均值为 0，标准差为 0.02 的正态分布初始化权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果有填充索引，则将对应的权重设置为 0


class SimpleEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.apply(self._init_weights)

    def forward(self, sequence):
        x = self.token(sequence)
        return self.dropout(x)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


















'''
baibai
'''




#
#
# class BaiEmbedding(nn.Module):
#
#     def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
#         """
#         :param vocab_size: total vocab size
#         :param embed_size: embedding size of token embedding
#         :param dropout: dropout rate
#         """
#         num_items=vocab_size-5
#         self.item_embedding = nn.Embedding(num_items, embed_size)
#         nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
#         # 创建词嵌入层，vocab_size 是词汇表大小，embed_size 是嵌入向量维度，padding_idx 是填充索引
#
#
#         self.W = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_layer)])
#         for w in self.W:
#             nn.init.normal_(w.weight, mean=0, std=0.01)
#         # 这里创建了一个包含多个线性层的列表 self.W，用于多层的图神经网络。
#         # 每个线性层的输入和输出维度都是 num_factor，这是项目的嵌入维度。
#         # 通过 nn.init.normal_ 方法初始化这些线性层的权重。
#
#
#         self.position_embedding = nn.Embedding(max_len, embed_size)
#         nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)
#
#
#         self.dropout = nn.Dropout(p=dropout)
#         # 创建 dropout 层，dropout 参数为丢弃率
#         self.embed_size = embed_size
#         # 保存嵌入向量的维度
#         self.apply(self._init_weights)
#         # 应用 _init_weights 函数初始化权重
#
#         num_layer=2
#
#
#
#
#
#     def forward(self, sequence):
#         adj_matrix=sequence[1]
#         input_seq=sequence[0]
#         adj_matrix_dropout = self.node_dropout(adj_matrix, len(adj_matrix[0]), 1 - self.dropout)
#         item_embedding_final = [self.item_embedding.weight]
#         layer = self.item_embedding(input_seq)
#
#         for k in range(self.num_layer):
#             layer = torch.matmul(torch.sparse.mm(adj_matrix_dropout, layer), self.W[k].weight)
#             layer = F.tanh(layer)
#             item_embedding_final.append(layer)
#
#         item_embedding_final = torch.mean(torch.stack(item_embedding_final, dim=1), dim=1)#所有层的嵌入进行求平均
#
#         position = torch.arange(self.max_len).unsqueeze(0).repeat(input_seq.size(0), 1)
#         p_emb = self.position_embedding(position)
#
#
#         seq = item_embedding_final.index_select(0, input_seq) * (self.num_factor ** 0.5) + p_emb
#
#         return self.dropout(seq)
#         # 应用 dropout 并返回最终嵌入结果
#
#
#     def _init_weights(self, module):
#         """Initialize the weights."""
#         # 初始化权重的函数
#         if isinstance(module, nn.Embedding):
#             # 如果模块是嵌入层
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             # 使用均值为 0，标准差为 0.02 的正态分布初始化权重
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#                 # 如果有填充索引，则将对应的权重设置为 0
# 
#         # if self.gnn_type == 'gcn':
#         #     for k in range(self.num_layer):
#         #         layer = torch.matmul(torch.sparse.mm(adj_matrix_dropout, layer), self.W[k].weight)
#         #         layer = F.tanh(layer)
#         #         item_embedding_final.append(layer)
#         # elif self.gnn_type == 'sgc':
#         #     for k in range(self.num_layer):
#         #         layer = torch.sparse.mm(adj_matrix_dropout, layer)
#         #         item_embedding_final.append(layer)
#         # else:
#         #     pass
#
#         # if self.layer_agg_type == 'sum':
#         #     item_embedding_final = torch.sum(torch.stack(item_embedding_final, dim=1), dim=1)
#         # elif self.layer_agg_type == 'avg':
#         #     item_embedding_final = torch.mean(torch.stack(item_embedding_final, dim=1), dim=1)
#         # elif self.layer_agg_type == 'concat':
#         #     item_embedding_final = torch.cat(item_embedding_final, dim=1)
#         #     self.num_factor *= (self.num_layer + 1)
#         # else:
#         #     item_embedding_final = item_embedding_final[-1]
#
