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


class LossInDefender(Defender):
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

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):
        if len(poison_data['train']) > 100000:      # 待修正
            self.lr = 4e-6
        else:
            self.lr = 2e-4

        noise_data = copy.deepcopy(poison_data)
        noise_data = add_data_noise(noise_data, 30)
        weights = self.predetect(model=model, poison_data=noise_data)

        # 直接使用weight调整训练权重
        new_train = []
        for ii, orig_tuple in enumerate(poison_data['train']):
            new_train.append(orig_tuple + (weights[ii],))
        poison_data['train'] = new_train

        model = self.trainer.train(model, poison_data)
        return model

    def predetect(self, model: Optional[Victim] = None,
                poison_data: Optional[Dict] = None):
        count = 0
        dbi = 1000
        dic = {'dbi':dbi}
        while dbi > 0.45 and count < 5:
            dataloader = wrap_dataset(poison_data, self.batch_size, shuffle=True)

            model2 = copy.deepcopy(model)
            self.basetrainer = load_trainer(dict(self.train_config, **{"name": "base", "visualize": True, "lr": self.basetrainer_lr}))
            self.basetrainer.register(model2, dataloader, ["accuracy"])

            loss_list1, confidence_list1 = self.basetrainer.loss_one_epoch(0, poison_data)
            self.basetrainer.train_one_epoch(0, poison_data)
            loss_list2, confidence_list2 = self.basetrainer.loss_one_epoch(1, poison_data)

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
            plot = False
            if plot:
                sns.displot(data=df, x='dc', hue='ltrue', palette=sns.color_palette("hls", 8))
                plt.title('dc')
                plt.show()

                sns.displot(data=df, x='dl', hue='ltrue', palette=sns.color_palette("hls", 8))
                plt.title('dl')
                plt.show()

            # 特性1 poison的label的dc特别大
            th = np.percentile(dc, (1-self.threshold) * 100)        # 找到dc的分为点
            max_ltrue = df[df.dc > th]['ltrue'].values          # 找到dc小的数据
            counts_dc = np.bincount(max_ltrue)                  # 统计这些数据中不同类别数据的个数
            pred_target_label_dc = np.argmax(counts_dc)         # 找到dc小的label中个数最多的数据
            rate_dc = counts_dc[pred_target_label_dc]/np.sum(counts_dc) # 查看预测标签在所有label中的占比

            print('dc pred:%f, confidence: %f' % (
            pred_target_label_dc, rate_dc))

            pred_target_label = pred_target_label_dc
            df_poison = df[df.ltrue == pred_target_label]

            if plot:
                sns.displot(data=df_poison, x='dc', hue='lpoison', palette=sns.color_palette("hls", 8))
                plt.show()

                sns.displot(data=df_poison, x='dl', hue='lpoison', palette=sns.color_palette("hls", 8))
                plt.show()

            # Step2 发现poison data，dc小的为poison datac
            data = df_poison['dc'].values
            scaler = MinMaxScaler()
            scaled_data_reshaped = scaler.fit_transform(data.reshape(-1, 1)).flatten().reshape(-1, 1)

            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
            gmm.fit(scaled_data_reshaped)

            labels = gmm.predict(scaled_data_reshaped)  # 获得预测的类别
            label_prob = gmm.predict_proba(scaled_data_reshaped)  # 获得预测的类别的概率

            # 调整label，保证dc大的数据的label为1-中毒
            means = gmm.means_
            larger_mean_index = np.argmax(means[:, 0])

            if larger_mean_index == 1:
                poison_prob = label_prob[:, 1]
            else:
                poison_prob = label_prob[:, 0]

            # 判断拟合结果好坏
            dbi = davies_bouldin_score(scaled_data_reshaped, labels)
            print(dbi)
            if plot:
                x = np.linspace(0, 1, 1000).reshape(-1, 1)
                logprob = gmm.score_samples(x)
                pdf = np.exp(logprob)

                plt.plot(x, pdf, '-r', label='GMM fit')
                plt.hist(scaled_data_reshaped, bins=30, density=True, alpha=0.5, color='blue',
                 label='Histogram of dl (lpoison=0)')
                plt.show()

            count += 1
            if dic['dbi'] > dbi:
                dic['dbi'] = dbi
                dic['prob'] = poison_prob
                dic['pred_target_label'] = pred_target_label
                df.to_csv('./loss/%s.csv' % self.path)

        dbi = dic['dbi']
        prob = dic['prob']
        pred_target_label = dic['pred_target_label']

        self.info = 'davies_bouldin_score-%f' % dbi

        df = pd.read_csv('./loss/%s.csv' % self.path)

        # 平滑正负
        prob = -1.6 * prob + 0.8
        weights = 1 / 2 * np.log((1 + prob) / (1 - prob))

        df_poison = df[df.ltrue == pred_target_label]
        df_poison['weight'] = weights
        df = pd.merge(df, df_poison[['index', 'weight']], on='index', how='left')
        df['weight'].fillna(1, inplace=True)
        weights = df['weight'].values

        return weights


def minmax(loss):
    __min__ = np.min(loss)-0.01
    __max__ = np.max(loss)+0.01
    if __min__ != __max__:
        loss = (loss - __min__)/(__max__ - __min__)
    else:
        loss = np.zeros(len(loss))
    return loss