

from .base import AbstractDataloader

import torch
import numpy as np
import torch.utils.data as data_utils

class RecDataloader(AbstractDataloader):
    def __init__(
            self,
            dataset,
            seg_len,
            mask_prob,
            num_items,
            num_workers,
            val_negative_sampler_code,
            val_negative_sample_size,
            train_batch_size,
            val_batch_size,
            predict_only_target=False,
        ):
        super().__init__(dataset,
            val_negative_sampler_code,
            val_negative_sample_size)
        self.target_code = self.bmap.get('buy') if self.bmap.get('buy') else self.bmap.get('pos')
        self.seg_len = seg_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_only_target = predict_only_target

    # 定义了一个方法，用于获取训练数据集的 DataLoader
    def get_train_loader(self):
        # 获取训练数据集对象
        dataset = self._get_train_dataset()
        # 使用 DataLoader 将数据集分成批次进行加载
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def get_test_loader(self):
        dataset = self._get_eval_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = RecTrainDataset(self.train, self.train_b, self.seg_len, self.mask_prob, self.num_items, self.target_code, self.predict_only_target)
        return dataset

    def get_val_loader(self):
        dataset = self._get_eval_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=self.num_workers)
        return dataloader

    def _get_eval_dataset(self):
        dataset = RecEvalDataset(self.train, self.train_b, self.val, self.val_b, self.val_num, self.seg_len, self.num_items, self.target_code, self.val_negative_samples)
        return dataset

# 定义了一个数据集类，用于推荐系统的训练数据集
class RecTrainDataset(data_utils.Dataset):
    # 初始化方法，接受一些参数来构建数据集
    def __init__(self, u2seq, u2b, max_len, mask_prob, num_items, target_code, predict_only_target):
        # 初始化数据集的一些属性
        self.u2seq = u2seq  # 用户到序列的映射
        self.u2b = u2b  # 用户到行为的映射
        self.users = sorted(self.u2seq.keys())  # 用户列表，按照键排序
        self.max_len = max_len  # 序列的最大长度
        self.mask_prob = mask_prob  # 掩码概率
        self.num_items = num_items  # 项目数量
        self.target_code = target_code  # 目标代码
        self.predict_only_target = predict_only_target  # 仅预测目标标记
    # 返回数据集的长度
    def __len__(self):
        return len(self.users)
    # 获取数据集中特定索引的数据项
    def __getitem__(self, index):
        # 获取特定索引对应的用户
        user = self.users[index]
        # 获取用户的序列和行为
        seq = self.u2seq[user]
        b_seq = self.u2b[user]
        # 初始化列表来存储tokens，行为和标签
        tokens = []
        behaviors = []
        labels = []
        # 遍历序列和行为
        for s, b in zip(seq, b_seq):
            # 生成一个随机概率
            prob = np.random.rand()
            # 如果随机概率小于掩码概率且不仅预测目标标记
            if prob < self.mask_prob and not self.predict_only_target:
                # 将特殊标记添加到tokens列表中
                tokens.append(self.num_items + 1)
                # 将标签添加到labels列表中
                labels.append(s)
            # 如果随机概率小于掩码概率且仅预测目标标记且行为等于目标代码
            elif prob < self.mask_prob and self.predict_only_target and b == self.target_code:
                # 将特殊标记添加到tokens列表中
                tokens.append(self.num_items + 1)
                # 将标签添加到labels列表中
                labels.append(s)
            else:
                # 否则，将序列添加到tokens列表中
                tokens.append(s)
                # 添加0作为标签
                labels.append(0)
            # 将行为添加到behaviors列表中
            behaviors.append(b)
        # 如果tokens列表的长度小于等于最大长度或者随机概率小于0.8
        if len(tokens) <= self.max_len or np.random.rand() < 0.8:
            # 截断tokens、labels和behaviors列表为最大长度
            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            behaviors = behaviors[-self.max_len:]
            # 计算填充长度
            padding_len = self.max_len - len(tokens)
            # 填充tokens、labels和behaviors列表
            tokens = [0] * padding_len + tokens
            labels = [0] * padding_len + labels
            behaviors = [0] * padding_len + behaviors
        else:
            # 随机截取tokens、labels和behaviors列表为最大长度
            begin_idx = np.random.randint(0, len(tokens) - self.max_len + 1)
            tokens = tokens[begin_idx:begin_idx + self.max_len]
            labels = labels[begin_idx:begin_idx + self.max_len]
            behaviors = behaviors[begin_idx:begin_idx + self.max_len]
        # 返回字典，包含输入的ids、标签和行为
        return {
            'input_ids': torch.LongTensor(tokens),
            'labels': torch.LongTensor(labels),
            'behaviors': torch.LongTensor(behaviors)
        }


'''
bai
'''
# # 定义了一个数据集类，用于推荐系统的训练数据集
# class RecTrainDataset(data_utils.Dataset):
#     # 初始化方法，接受一些参数来构建数据集
#     def __init__(self, u2seq, u2b, max_len, mask_prob, num_items, target_code, predict_only_target):
#         # 初始化数据集的一些属性
#         self.u2seq = u2seq  # 用户到序列的映射
#         self.u2b = u2b  # 用户到行为的映射
#         self.users = sorted(self.u2seq.keys())  # 用户列表，按照键排序
#         self.max_len = max_len  # 序列的最大长度
#         self.mask_prob = mask_prob  # 掩码概率
#         self.num_items = num_items  # 项目数量
#         self.target_code = target_code  # 目标代码
#         self.predict_only_target = predict_only_target  # 仅预测目标标记
#
#         # self.adj_matrix = adj_matrix  # 加载邻接矩阵
#
#
#     # 返回数据集的长度
#     def __len__(self):
#         return len(self.users)
#     # 获取数据集中特定索引的数据项
#     def __getitem__(self, index):
#         # 获取特定索引对应的用户
#         user = self.users[index]
#         # 获取用户的序列和行为
#         seq = self.u2seq[user]
#         b_seq = self.u2b[user]
#         # 初始化列表来存储tokens，行为和标签
#         tokens = []
#         behaviors = []
#         labels = []
#         adj_matrix = [][]
#         # 更新邻接矩阵中的对应元素
#         for s, b in zip(seq, b_seq):
#             # 更新邻接矩阵中的对应元素
#             adj_matrix[user, s] = 1
#
#             prob = np.random.rand()
#             if prob < self.mask_prob and not self.predict_only_target:
#                 tokens.append(self.num_items + 1)
#                 labels.append(s)
#             elif prob < self.mask_prob and self.predict_only_target and b == self.target_code:
#                 tokens.append(self.num_items + 1)
#                 labels.append(s)
#             else:
#                 tokens.append(s)
#                 labels.append(0)
#             behaviors.append(b)
#
#
#
#
#
#
#         # 遍历序列和行为
#         for s, b in zip(seq, b_seq):
#             # 生成一个随机概率
#             prob = np.random.rand()
#             # 如果随机概率小于掩码概率且不仅预测目标标记
#             if prob < self.mask_prob and not self.predict_only_target:
#                 # 将特殊标记添加到tokens列表中
#                 tokens.append(self.num_items + 1)
#                 # 将标签添加到labels列表中
#                 labels.append(s)
#             # 如果随机概率小于掩码概率且仅预测目标标记且行为等于目标代码
#             elif prob < self.mask_prob and self.predict_only_target and b == self.target_code:
#                 # 将特殊标记添加到tokens列表中
#                 tokens.append(self.num_items + 1)
#                 # 将标签添加到labels列表中
#                 labels.append(s)
#             else:
#                 # 否则，将序列添加到tokens列表中
#                 tokens.append(s)
#                 # 添加0作为标签
#                 labels.append(0)
#             # 将行为添加到behaviors列表中
#             behaviors.append(b)
#         # 如果tokens列表的长度小于等于最大长度或者随机概率小于0.8
#         if len(tokens) <= self.max_len or np.random.rand() < 0.8:
#             # 截断tokens、labels和behaviors列表为最大长度
#             tokens = tokens[-self.max_len:]
#             labels = labels[-self.max_len:]
#             behaviors = behaviors[-self.max_len:]
#             # 计算填充长度
#             padding_len = self.max_len - len(tokens)
#             # 填充tokens、labels和behaviors列表
#             tokens = [0] * padding_len + tokens
#             labels = [0] * padding_len + labels
#             behaviors = [0] * padding_len + behaviors
#         else:
#             # 随机截取tokens、labels和behaviors列表为最大长度
#             begin_idx = np.random.randint(0, len(tokens) - self.max_len + 1)
#             tokens = tokens[begin_idx:begin_idx + self.max_len]
#             labels = labels[begin_idx:begin_idx + self.max_len]
#             behaviors = behaviors[begin_idx:begin_idx + self.max_len]
#         # 返回字典，包含输入的ids、标签和行为
#         # return {
#         #     'input_ids': torch.LongTensor(tokens),
#         #     'labels': torch.LongTensor(labels),
#         #     'behaviors': torch.LongTensor(behaviors),
#         #     'adj_matrix': torch.FloatTensor(self.adj_matrix)  # 返回邻接矩阵
#         # }
#         return [
#             {
#                 'input_ids': torch.LongTensor(tokens),
#                 'labels': torch.LongTensor(labels),
#                 'behaviors': torch.LongTensor(behaviors)
#             },
#             torch.FloatTensor(self.adj_matrix)  # 返回邻接矩阵
#         ]
#         # return  torch.FloatTensor(self.adj_matrix)  # 返回邻接矩阵
#
#

'''
bai
'''









class RecEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2b, u2answer, u2ab, val_num, max_len, num_items, target_code, negative_samples):
        self.u2seq = u2seq
        self.u2b = u2b
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.u2ab = u2ab
        self.val_num = val_num
        self.max_len = max_len
        self.negative_samples = negative_samples
        self.num_items = num_items
        self.target_code = target_code

    def __len__(self):
        return self.val_num

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.num_items + 1]
        seq = seq[-self.max_len:]
        seq_b = self.u2b[user] + self.u2ab[user]
        seq_b = seq_b[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        seq_b = [0] * padding_len + seq_b

        return {
            'input_ids':torch.LongTensor(seq),
            'candidates':torch.LongTensor(candidates), 
            'labels':torch.LongTensor(labels),
            'behaviors': torch.LongTensor(seq_b)
        }