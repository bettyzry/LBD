from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from .trainer import Trainer
from transformers import  get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
from typing import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class MLMTrainer(Trainer):
    def __init__(
        self,
        batch_size: Optional[int] = 32,
        epochs: Optional[int] = 5,
        lr: Optional[float] = 2e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.path = '/home/server/Documents/zry/OpenBackdoor-main/models/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.model = BertForMaskedLM.from_pretrained(self.path)

    def register(self, model, dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        train_length = len(dataloader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=self.warm_up_epochs * train_length,
                                                         num_training_steps=self.epochs * train_length)
        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

    def loss_one_epoch(self, epoch: int, dataset):
        dataset = TextDataset(dataset, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()  # 切换模型到评估模式
        loss_list = []

        # 迭代每个批次
        for batch_idx, (input_ids, labels) in enumerate(tqdm(dataloader, desc="Iteration")):
            with torch.no_grad():  # 禁用梯度计算，因为不需要训练
                # 前向传播计算 logits
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

                batch_loss = []

                # 对于批次中的每个样本，计算单独的损失
                for i in range(input_ids.size(0)):  # 遍历批次中的每个样本
                    # 获取 [MASK] 的位置
                    mask_token_index = torch.where(input_ids[i] == self.tokenizer.mask_token_id)[0]

                    if len(mask_token_index) > 0:  # 确保存在 [MASK] 标记
                        # 获取该位置的预测结果和真实标签
                        mask_token_logits = logits[i, mask_token_index, :]
                        mask_token_labels = labels[i, mask_token_index]

                        # 计算每个样本的损失
                        loss = F.cross_entropy(mask_token_logits, mask_token_labels)
                        batch_loss.append(loss.item())
                    else:
                        # 如果没有 [MASK]，则损失为 0
                        batch_loss.append(0.0)
                loss_list += batch_loss

        loss_list = np.array(loss_list)
        return loss_list

    def train_one_epoch(self, epoch: int, train_dataloader):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.

        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.model.train()
        total_loss = 0
        for batch_idx, (input_ids, labels) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            self.optimizer.zero_grad()  # 清空梯度

            # 前向传播计算 logits
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # 反向传播计算梯度
            loss.backward()

            # 更新模型参数
            self.optimizer.step()
            self.scheduler.step()

            # 统计损失
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        return avg_loss

    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"]):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """
        dataset = dataset['train']
        dataset_text = [i[0] for i in dataset]
        dataloader = DataLoader(TextDataset(dataset_text, self.tokenizer), batch_size=self.batch_size, shuffle=True)

        self.register(model, dataloader, metrics)

        if self.visualize:
            self.info['data'] = [d[0] for i, d in enumerate(dataset)]
            self.info['ltrue'] = [d[1] for i, d in enumerate(dataset)]
            self.info['lpoison'] = [d[2] for i, d in enumerate(dataset)]
        for epoch in range(self.epochs):
            if self.visualize:
                loss_list = self.loss_one_epoch(epoch, dataset_text)
                self.info['l_%s'%epoch] = loss_list
            epoch_loss = self.train_one_epoch(epoch, dataloader)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        return self.model

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 对句子进行标记化，并生成掩码
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                truncation=True)
        labels = inputs.input_ids.detach().clone()

        # 创建掩码 (随机掩盖 15% 的词)
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != self.tokenizer.cls_token_id) * \
                   (inputs.input_ids != self.tokenizer.sep_token_id) * (inputs.input_ids != self.tokenizer.pad_token_id)

        inputs.input_ids[mask_arr] = self.tokenizer.mask_token_id

        return inputs.input_ids.squeeze(), labels.squeeze()