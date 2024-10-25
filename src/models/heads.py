

import torch
import torch.nn as nn
import torch.nn.functional as F

# head used for bert4rec
class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""
    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1
        self.out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        self.ln = nn.LayerNorm(d_model)
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, b_seq, candidates=None):
        x = self.out(x)  # B x H or M x H
        x = self.ln(x)
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V

        # return logits
        return logits



class CGCDotProductPredictionHead(nn.Module):
    """
    具有共享专家和行为特定专家的模型
    3个共享专家，
    每个行为一个特定专家。
    """

    def __init__(self, d_model, n_b, n_e_sh, n_e_sp, num_items, token_embeddings):
        super().__init__()
        self.n_b = n_b  # 设置行为的数量
        self.n_e_sh = n_e_sh  # 共享专家的数量
        self.n_e_sp = n_e_sp  # 行为特定专家的数量
        self.vocab_size = num_items + 1  # 词汇表大小
        self.softmax = nn.Softmax(dim=-1)  # softmax层
        # 定义共享专家列表，每个专家都是一个包含线性层的序列模块
        self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_e_sh)])
        # 定义行为特定专家列表，每个专家都是一个包含线性层的序列模块
        self.specific_experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_b * self.n_e_sp)])
        # 定义权重门参数，形状为(n_b, d_model, n_e_sh + n_e_sp)，用于控制每个专家的贡献
        self.w_gates = nn.Parameter(torch.randn(self.n_b, d_model, self.n_e_sh + self.n_e_sp), requires_grad=True)
        self.token_embeddings = token_embeddings  # 标记嵌入层
        self.ln = nn.LayerNorm(d_model)  # Layer Norm层

    def forward(self, x, b_seq, candidates=None):
        x = self.mmoe_process(x, b_seq)  # 处理输入数据
        if candidates is not None:  # 如果有候选项
            emb = self.token_embeddings(candidates)  # 获取候选项的嵌入表示
            logits = (x.unsqueeze(1) * emb).sum(-1)  # 计算与候选项的相似度得分
        else:  # 如果没有候选项
            emb = self.token_embeddings.weight[:self.vocab_size]  # 获取词汇表的嵌入表示
            logits = torch.matmul(x, emb.transpose(0, 1))  # 计算与词汇表中词的相似度得分
        return logits

    def mmoe_process(self, x, b_seq):
        shared_experts_o = [e(x) for e in self.shared_experts]  # 对输入数据应用共享专家
        specific_experts_o = [e(x) for e in self.specific_experts]  # 对输入数据应用行为特定专家
        gates_o = self.softmax(torch.einsum('nd,tde->tne', x, self.w_gates))  # 计算专家权重
        # 重新排列专家输出
        experts_o_tensor = torch.stack(
            [torch.stack(shared_experts_o + specific_experts_o[i * self.n_e_sp:(i + 1) * self.n_e_sp]) for i in
             range(self.n_b)])
        output = torch.einsum('tend,tne->tnd', experts_o_tensor, gates_o)  # 根据权重加权求和专家输出
        outputs = torch.cat([torch.zeros_like(x).unsqueeze(0), output])  # 将全零张量与输出连接起来
        return x + self.ln(torch.einsum('tnd, nt -> nd', outputs, F.one_hot(b_seq,
                                                                            num_classes=self.n_b + 1).float()))  # 加权和结果与输入数据进行残差连接，并应用Layer Norm



# class DotProductPredictionHead(nn.Module):
#     """share embedding parameters"""
#     def __init__(self, d_model, num_items, token_embeddings):
#         super().__init__()
#         self.token_embeddings = token_embeddings
#         self.vocab_size = num_items + 1

#     def forward(self, x, b_seq, candidates=None):
#         if candidates is not None:  # x : B x H
#             emb = self.token_embeddings(candidates)  # B x C x H
#             logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
#         else:  # x : M x H
#             emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
#             logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
#         return logits