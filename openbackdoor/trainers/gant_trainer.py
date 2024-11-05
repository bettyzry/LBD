from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import  AdamW, get_linear_schedule_with_warmup
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from typing import *
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from umap import UMAP
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class GantTrainer(object):
    r"""
    Basic clean trainer. Used in clean-tuning and dataset-releasing attacks.

    Args:
        name (:obj:`str`, optional): name of the trainer. Default to "Base".
        lr (:obj:`float`, optional): learning rate. Default to 2e-5.
        weight_decay (:obj:`float`, optional): weight decay. Default to 0.
        epochs (:obj:`int`, optional): number of epochs. Default to 10.
        batch_size (:obj:`int`, optional): batch size. Default to 4.
        gradient_accumulation_steps (:obj:`int`, optional): gradient accumulation steps. Default to 1.
        max_grad_norm (:obj:`float`, optional): max gradient norm. Default to 1.0.
        warm_up_epochs (:obj:`int`, optional): warm up epochs. Default to 3.
        ckpt (:obj:`str`, optional): checkpoint name. Can be "best" or "last". Default to "best".
        save_path (:obj:`str`, optional): path to save the model. Default to "./models/checkpoints".
        loss_function (:obj:`str`, optional): loss function. Default to "ce".
        visualize (:obj:`bool`, optional): whether to visualize the hidden states. Default to False.
        poison_setting (:obj:`str`, optional): the poisoning setting. Default to mix.
        poison_method (:obj:`str`, optional): name of the poisoner. Default to "Base".
        poison_rate (:obj:`float`, optional): the poison rate. Default to 0.1.

    """
    def __init__(
        self, 
        name: Optional[str] = "Base",
        lr: Optional[float] = 2e-5,
        weight_decay: Optional[float] = 0.,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 4,
        gradient_accumulation_steps: Optional[int] = 1,
        max_grad_norm: Optional[float] = 1.0,
        warm_up_epochs: Optional[int] = 3,
        ckpt: Optional[str] = "best",
        save_path: Optional[str] = "./checkpoints",
        loss_function: Optional[str] = "ce",
        visualize: Optional[bool] = False,
        poison_setting: Optional[str] = "mix",
        poison_method: Optional[str] = "Base",
        poison_rate: Optional[float] = 0.01,
        **kwargs):

        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warm_up_epochs = warm_up_epochs
        self.ckpt = ckpt

        timestamp = int(datetime.now().timestamp())
        self.save_path = os.path.join(save_path, f'{poison_setting}-{poison_method}-{poison_rate}', str(timestamp))
        os.makedirs(self.save_path, exist_ok=True)

        self.visualize = visualize
        self.poison_setting = poison_setting
        self.poison_method = poison_method
        self.poison_rate = poison_rate

        self.COLOR = ['royalblue', 'red', 'palegreen', 'violet', 'paleturquoise', 
                            'green', 'mediumpurple', 'gold', 'deepskyblue']

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.info = {}
        self.target = 0
        self.lambda_mse = 0.1
    
    def register(self, model: Victim, dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.classifier = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.classifier.train()
        self.classifier.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = int(len(dataloader["train"])/self.batch_size)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * train_length,
                                                    num_training_steps=self.epochs * train_length)
        
        self.poison_loss_all = []
        self.normal_loss_all = []
        reduction = "none" if self.visualize else "mean"
        self.loss_function = nn.CrossEntropyLoss(reduction=reduction)

        # 生成器
        embed_dim = self.classifier.plm.config.hidden_size
        self.generator = Generator(embed_dim).to(self.classifier.device)
        self.optimizer_g = AdamW(self.generator.parameters(), lr=self.lr)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

    def train_classifier(self, epoch: int, train_dataloader):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.
        
        Returns:
            :obj:`float`: average loss of the epoch.
        """
        self.generator.eval()
        self.classifier.train()

        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_inputs, batch_labels = self.classifier.process(batch)
            input_ids = batch_inputs['input_ids']
            attention_mask = batch_inputs['attention_mask']
            # if len(batch_inputs.data['input_ids'][1]) > 512:
            #     i = 1

            with torch.no_grad():
                original_embeddings = self.classifier.plm.bert.embeddings(input_ids)
            # 生成器生成新的embedding'
            generated_embeddings = self.generator(original_embeddings)
            # 使用生成的embedding进行判别
            output = self.classifier({'inputs_embeds': generated_embeddings,
                                      'attention_mask': attention_mask})

            # output = self.classifier(batch_inputs)
            logits = output.logits
            loss = nn.CrossEntropyLoss(reduction='mean')(logits, batch_labels)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.classifier.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        logger.info('Epoch: {}, avg loss: {}'.format(epoch + 1, avg_loss))

    def train_generator(self, epoch, train_dataloader):
        self.generator.train()
        self.classifier.eval()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            output, ytrue = self.classifier.process(batch)
            input_ids = output['input_ids']
            attention_mask = output['attention_mask']
            ytarget = torch.full_like(ytrue, self.target, device=self.classifier.device)

            with torch.no_grad():
                original_embeddings = self.classifier.plm.bert.embeddings(input_ids)

            # 生成器生成新的embedding'
            generated_embeddings = self.generator(original_embeddings)

            # 将生成的embedding传回分类器并计算新的输出
            classifier_outputs = self.classifier({'inputs_embeds': generated_embeddings,
                                                  'attention_mask': attention_mask}).logits

            # 计算损失
            loss_ce_target = self.criterion_ce(classifier_outputs, ytarget)  # 针对目标标签的分类损失
            loss_mse = self.criterion_mse(generated_embeddings, original_embeddings)  # 重建损失

            # 总损失
            loss_g = loss_ce_target + self.lambda_mse * loss_mse
            total_loss += loss_g

            # 优化生成器
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

        print(f"Epoch [{epoch + 1}], Loss Generator: {total_loss/len(train_dataloader):.4f}")

    def evaluate(self, model, eval_dataloader, metrics, plot=False, info=''):
        """
        Evaluate the model.

        Args:
            model (:obj:`Victim`): victim model.
            eval_dataloader (:obj:`torch.utils.data.DataLoader`): dataloader for evaluation.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].

        Returns:
            results (:obj:`Dict`): evaluation results.
            dev_score (:obj:`float`): dev score.
        """
        results, dev_score = evaluate_classification(model, eval_dataloader, metrics, plot=plot, info=info)
        return results, dev_score


class Generator(nn.Module):
    def __init__(self, embedding_size, hidden_dim=1024):
        super(Generator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_size)
        )

    def forward(self, x):
        return self.mlp(x)
