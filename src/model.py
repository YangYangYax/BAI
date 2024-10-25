
import torch
import pytorch_lightning as pl
from .models import CGCDotProductPredictionHead, DotProductPredictionHead
from .models.bert4rec import BERT
from .utils import recalls_and_ndcgs_for_ks


class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: BERT,
            b_head: bool = False,
        ):
        super().__init__()
        self.backbone = backbone
        self.n_b = backbone.n_b
        if b_head:
        # if False:
            self.head = CGCDotProductPredictionHead(backbone.d_model, self.n_b, 3, 1, backbone.num_items, self.backbone.embedding.token)
        else:
            self.head = DotProductPredictionHead(backbone.d_model, backbone.num_items, self.backbone.embedding.token)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, b_seq):
        return self.backbone(input_ids, b_seq)#调用bert的forword，所以input_ids是x
    '''
    
    在 RecTrainDataset 类中，__getitem__ 方法定义了如何获取数据集中的一个样本。
    在这个方法中，input_ids 是通过以下步骤生成的：
    
    首先从数据集中获取一个用户的序列 seq 和对应的行为序列 b_seq。
    遍历该用户的序列和行为序列，对每个元素进行处理：
    如果随机数小于 mask_prob 并且不仅预测目标，或者如果随机数小于 mask_prob 并且只预测目标，
    并且当前行为等于目标代码，则将一个特定的值（num_items + 1）添加到 tokens 和 labels 中，
    表示进行了mask操作。
    否则，将原始的序列元素添加到 tokens 中，对应的标签为0。
    根据是否需要截断序列或者随机截取一段序列，对 tokens 进行处理得到 input_ids。
    返回一个包含 input_ids、labels 和 behaviors 的字典。
    因此，input_ids 是根据原始序列以及一定的概率策略生成的，用于表示输入文本序列的token序列。
    
    '''

    def training_step(self, batch, batch_idx):
        # 从批次中获取输入序列的input_ids
        input_ids = batch['input_ids']

        # 从批次中获取行为序列的b_seq
        b_seq = batch['behaviors']

        # 使用模型进行前向传播，计算模型的输出
        outputs = self(input_ids, b_seq)

        # 调整模型输出的形状以准备计算损失
        outputs = outputs.view(-1, outputs.size(-1))

        # 从批次中获取真实标签 labels
        labels = batch['labels']

        # 调整真实标签的形状以匹配模型输出的形状
        labels = labels.view(-1)

        # 创建一个布尔张量，标记哪些位置的标签是有效的（非零的）
        valid = labels > 0

        # 找到所有有效标签的索引
        valid_index = valid.nonzero().squeeze()

        # 提取所有有效位置的模型输出
        valid_outputs = outputs[valid_index]

        # 提取所有有效位置的行为序列
        valid_b_seq = b_seq.view(-1)[valid_index]

        # 提取所有有效位置的真实标签
        valid_labels = labels[valid_index]

        # 使用模型头部对有效位置的模型输出和行为序列进行预测
        valid_logits = self.head(valid_outputs, valid_b_seq)

        # 使用预测结果和真实标签计算损失
        loss = self.loss(valid_logits, valid_labels)

        # 增加一个维度，以匹配PyTorch Lightning中期望的损失格式
        loss = loss.unsqueeze(0)

        # 返回一个字典，包含损失值，符合PyTorch Lightning的返回格式
        return {'loss': loss}
        
    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        outputs = self(input_ids, b_seq)

        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        last_b_seq = b_seq[:,-1]
        candidates = batch['candidates'].squeeze() # B x C
        logits = self.head(last_outputs, last_b_seq, candidates)
        labels = batch['labels'].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

        return metrics

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        outputs = self(input_ids, b_seq)

        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        last_b_seq = b_seq[:,-1]
        candidates = batch['candidates'].squeeze() # B x C
        logits = self.head(last_outputs, last_b_seq, candidates)
        labels = batch['labels'].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

        return metrics

    # def validation_epoch_end(self, validation_step_outputs):
    #     keys = validation_step_outputs[0].keys()
    #     for k in keys:
    #         tmp = []
    #         for o in validation_step_outputs:
    #             tmp.append(o[k])
    #         self.log(f'Val:{k}', torch.Tensor(tmp).mean())

    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            avg_metric = torch.Tensor(tmp).mean()
            print(f'Validation {k}: {avg_metric}')  # Print the average metric
            self.log(f'Val:{k}', avg_metric)  # Log the average metric to TensorBoard