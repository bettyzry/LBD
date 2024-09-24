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

class LossInTrainer(object):
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
    
    def register(self, model: Victim, dataloader, metrics):
        r"""
        Register model, dataloader and optimizer
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = self.metrics[0]
        self.split_names = dataloader.keys()
        self.model.train()
        self.model.zero_grad()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * train_length,
                                                    num_training_steps=self.epochs * train_length)
        
        self.poison_loss_all = []
        self.normal_loss_all = []
        self.loss_function = nn.CrossEntropyLoss(reduction="none")
        self.adj_loss = CustomLoss()

        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

    def train_one_epoch(self, epoch: int, epoch_iterator):
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
        poison_loss_list, normal_loss_list = [], []
        loss_list = []
        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function(logits, batch_labels)

            if self.visualize:
                loss_list.append(loss)
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())

            # 根据预先区分的set
            weight = batch['weight'].to('cuda')
            loss = loss * weight

            loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0
        if self.visualize:
            loss_list = torch.cat(loss_list).data.cpu().numpy()
        return avg_loss, avg_poison_loss, avg_normal_loss, loss_list

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
        dataloader = wrap_dataset(dataset, self.batch_size)

        train_dataloader = dataloader["train"]

        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        self.register(model, dataloader, metrics)

        best_dev_score = 0
        if self.visualize:
            self.info['data'] = [d[0] for i, d in enumerate(dataset['train'])]
            self.info['ltrue'] = [d[1] for i, d in enumerate(dataset['train'])]
            self.info['lpoison'] = [d[2] for i, d in enumerate(dataset['train'])]
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            epoch_loss, poison_loss, normal_loss, loss_list = self.train_one_epoch(epoch, epoch_iterator)
            if self.visualize:
                self.info['l_%s'%epoch] = loss_list
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)

            # if dev_score > best_dev_score:
            #     best_dev_score = dev_scorekaak
            #     if self.ckpt == 'best':
            #         torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        if self.visualize:
            self.save_vis()

        # if self.ckpt == 'last':
        #     torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        # logger.info("Training finished.")
        # state_dict = torch.load(self.model_checkpoint(self.ckpt))
        # self.model.load_state_dict(state_dict)
        # test_score = self.evaluate_all("test")
        return self.model

    def evaluate(self, model, eval_dataloader, metrics):
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
        results, dev_score = evaluate_classification(model, eval_dataloader, metrics)
        return results, dev_score

    def save_vis(self):
        df = pd.DataFrame(self.info)
        path = os.path.join('./info', '%s-%s-%s.csv' %
                            (self.poison_setting, self.poison_method, str(self.poison_rate)))
        df.to_csv(path, index=False)
        return

    def model_checkpoint(self, ckpt: str):
        return os.path.join(self.save_path, f'{ckpt}.ckpt')


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, loss, y2):
        # 计算 y2 == 0 和 y2 == 1 的掩码
        mask_0 = (y2 == 0).float()
        mask_1 = (y2 == 1).float()

        # 合并损失
        loss = loss*mask_0 - loss*mask_1
        return loss