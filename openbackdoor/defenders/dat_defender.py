import copy
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from .defender import Defender
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
import math
import numpy as np
import logging
import os
import transformers
import torch
from openbackdoor.victims import Victim
from tqdm import tqdm
from torch.utils.data import DataLoader
from openbackdoor.data import load_dataset, get_dataloader
import torch.nn as nn
from openbackdoor.data import get_dataloader, wrap_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from openbackdoor.trainers import load_trainer
from openbackdoor.trainers import Trainer
from openbackdoor.utils.add_noise import add_data_noise, add_label_noise, remove_words_from_text
from sklearn.mixture import GaussianMixture
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class DATDefender(Defender):
    r"""
        Defender for `ONION <https://arxiv.org/abs/2011.10369>`_

    Args:
        parallel (`bool`, optional): identify whether to use multiple gpus.
        threshold (`int`, optional): threshold to remove suspicious words.
        batch_size (`int`, optional): batch size of GPTLM.
    """

    def __init__(
        self, 
        parallel: Optional[bool] = True, 
        threshold: Optional[int] = 0, 
        batch_size: Optional[int] = 32,
        train=None,     # 训练集参数
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.batch_size = batch_size
        self.train = True
        self.train_config = train
        self.basetrainer_lr = 2e-4
        self.basetrainer = load_trainer(dict(self.train_config, **{"name": "base", "visualize": True, "lr": self.basetrainer_lr}))
        self.trainer = load_trainer(train)
        self.path = ''
        self.info = ''
        self.lr = 2e-4
        self.epoch = 5
        self.dt = None
        self.model = None

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):
        # step0： 找到target label，以及所有非target label的数据用于后续训练
        target_label = self.get_target_label(poison_data, model)
        ltrue = np.array([i[1] for i in poison_data['train']])
        no_target_index = np.where(ltrue != target_label)[0]
        no_target_train = [(item[0], 1, 0) for item in poison_data['train'][no_target_index]]

        self.register(model)

        # step1: 先正常进行训练
        self.model = self.trainer.train(self.model, poison_data)

        # step2: 训练触发器生成模型
        self.train_dt(no_target_train, model)

        # step3： 训练更正后的模型
        new_train = self.dt_generate(poison_data)
        new_train = [(item, ltrue[ii], 1) for ii, item in enumerate(new_train)]
        poison_data['train'] = new_train
        self.model = self.trainer.train(self.model, poison_data)
        return self.model

    def dt_generate(self, poison_data):
        train_dataloader = wrap_dataset(poison_data['train'], self.batch_size, shuffle=False)
        self.dt.eval()
        new_train = []
        for batch in train_dataloader:
            text = batch['text']
            ntext = self.dt(text)
            new_train.append(ntext)
        new_train = torch.cat(new_train).data.cpu().numpy()
        return new_train

    def train_dt(self, no_target_train, model):
        # 训练触发器生成模型x
        train_dataloader = wrap_dataset(no_target_train, self.batch_size, shuffle=True)

        self.model.eval()
        self.dt.train()
        for e in range(self.epochs):
            total_loss = 0
            for batch in train_dataloader:
                text = batch["text"]
                labels = batch["label"]
                ntext = self.dt(text)          # 生成的新文本
                input_batch = self.model.tokenizer(ntext, padding=True, truncation=True, max_length=self.model.max_len,
                                             return_tensors="pt").to(self.model.device)
                pred_logits = self.model(input_batch).logits
                loss = nn.CrossEntropyLoss(reduction='mean')(pred_logits, labels)

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.dt.zero_grad()
            print("Epoch %d, Loss %.4f" % (e, total_loss))
        return

    def register(self, model):
        self.dt = DT(self.batch_size, model.device)
        self.model = model
        self.optimizer = AdamW(self.dt.named_parameters(), lr=self.lr)
        return

    def get_target_label(self, poison_data, model):
        return 1

    def get_target_label_late(self, poison_data, model):
        if len(poison_data['train']) > 100000:      # 待修正
            self.lr = 4e-6
        elif len(poison_data['train']) > 10000:
            self.lr = 2e-5
        else:
            self.lr = 2e-4

        noise_data = copy.deepcopy(poison_data)
        noise_data = add_data_noise(noise_data, 30)

        dataloader = wrap_dataset(noise_data, self.batch_size, shuffle=True)

        model2 = copy.deepcopy(model)
        self.basetrainer = load_trainer(
            dict(self.train_config, **{"name": "base", "visualize": True, "lr": self.basetrainer_lr}))
        self.basetrainer.register(model2, dataloader, ["accuracy"])

        loss_list1, confidence_list1 = self.basetrainer.loss_one_epoch(0, noise_data)
        self.basetrainer.train_one_epoch(0, noise_data)
        loss_list2, confidence_list2 = self.basetrainer.loss_one_epoch(1, noise_data)

        dl = loss_list1 - loss_list2
        dc = confidence_list2 - confidence_list1

        # Step1 发现target label
        df = pd.DataFrame()
        df['data'] = [i[0] for i in poison_data['train']]
        df['ltrue'] = [i[1] for i in poison_data['train']]
        df['lpoison'] = [i[2] for i in poison_data['train']]
        df['dl'] = dl
        df['dc'] = dc

        index = [i for i in range(len(df))]
        df['index'] = index

        # 特性1 poison的label的dc特别大
        th = np.percentile(dc, (1 - self.threshold) * 100)  # 找到dc的分为点
        max_ltrue = df[df.dc > th]['ltrue'].values  # 找到dc小的数据
        counts_dc = np.bincount(max_ltrue)  # 统计这些数据中不同类别数据的个数
        pred_target_label_dc = np.argmax(counts_dc)  # 找到dc小的label中个数最多的数据
        rate_dc = counts_dc[pred_target_label_dc] / np.sum(counts_dc)  # 查看预测标签在所有label中的占比

        print('dc pred:%f, confidence: %f' % (
            pred_target_label_dc, rate_dc))

        pred_target_label = pred_target_label_dc
        return pred_target_label


class DT(nn.Module):
    def __init__(self, batch_size, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device