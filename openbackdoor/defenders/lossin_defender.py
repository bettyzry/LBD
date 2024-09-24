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
        self.basetrainer = load_trainer(dict(train, **{"name": "base", "visualize": True, "lr": 2e-4}))
        self.trainer = load_trainer(train)

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):
        noise_data = self.add_noise(poison_data.copy(), 10)             # 生成一个插入噪声的数据
        poison_data = self.predetect(model=model, poison_data=noise_data)
        model = self.trainer.train(model, poison_data)
        return model

    def add_noise(self, poison_dataset: Dict, n: int):
        def shuffle_two_words(sentence):
            words = sentence.split()
            if len(words) < 2:
                return sentence
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '.join(words)

        def shuffle_adjacent_letters(sentence, n):
            sentence_list = list(sentence)
            adjacent_indices = [i for i in range(len(sentence_list) - 1) if
                                sentence_list[i].isalpha() and sentence_list[i + 1].isalpha()]
            if len(adjacent_indices) < n:
                print('Too short to shuffle')
                n = len(adjacent_indices)
            selected_indices = random.sample(adjacent_indices, n)
            for idx in selected_indices:
                sentence_list[idx], sentence_list[idx + 1] = sentence_list[idx + 1], sentence_list[idx]
            return ''.join(sentence_list)

        for ii, data in enumerate(poison_dataset['train']):
            poison_dataset['train'][ii] = (shuffle_adjacent_letters(data[0], n), data[1], data[2])
            # poison_dataset['train'][ii] = (shuffle_two_words(data[0]), data[1], data[2])
        return poison_dataset

    def predetect(self, model: Optional[Victim] = None,
                poison_data: Optional[Dict] = None):
        dataloader = wrap_dataset(poison_data, self.batch_size, shuffle=True)
        train_dataloader = dataloader["train"]

        model2 = copy.deepcopy(model)
        self.basetrainer.register(model2, dataloader, ["accuracy"])
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        poison_loss_list1, normal_loss_list1, loss_list1 = self.basetrainer.loss_one_epoch(0, poison_data)
        self.basetrainer.train_one_epoch(0, epoch_iterator)
        poison_loss_list2, normal_loss_list2, loss_list2 = self.basetrainer.loss_one_epoch(1, poison_data)

        dl = loss_list1 - loss_list2

        # Step1 发现target label
        df = pd.DataFrame()
        df['data'] = [i[0] for i in poison_data['train']]
        df['ltrue'] = [i[1] for i in poison_data['train']]
        df['lpoison'] = [i[2] for i in poison_data['train']]
        df['dl'] = dl

        # df_target = df[df.ltrue == 1]
        # sns.displot(data=df_target, x='dl', hue='lpoison', palette=sns.color_palette("hls", 8))
        # plt.show()

        th = np.percentile(dl, (1 - self.threshold) * 100)
        max_ltrue = df[df.dl > th]['ltrue'].values
        counts = np.bincount(max_ltrue)
        pred_target_label = np.argmax(counts)

        # Step2 发现poison data
        target_df = df[df.ltrue == pred_target_label]
        target_dl = target_df['dl'].values
        target_th1 = np.percentile(target_dl, 80)
        target_th2 = np.percentile(target_dl, 50)
        df['weight'] = df.apply(
            lambda row: -1 if row['ltrue'] == 1 and row['dl'] > target_th1 else
            0 if row['ltrue'] == 1 and target_th2 < row['dl'] <= target_th1 else
            1,
            axis=1
        )

        # th限制正负
        # index = np.where(dl > th)[0]
        # weights = np.ones(len(dl))
        # weights[index] = -1
        weights = df['weight'].values

        # 随机
        # import random
        # index = random.sample(range(0, len(poison_data['train'])), int(len(poison_data['train'])*self.threshold))
        # weights = np.ones(len(poison_data['train']))
        # weights[index] = -1

        # 平滑正负
        # weights = artanhx(dl, th)
        # weights = (loss_list-th)**3

        new_train = []
        lpoison = []
        for ii, orig_tuple in enumerate(poison_data['train']):
            # print(orig_tuple[2], dlabels[ii])
            new_train.append(orig_tuple + (weights[ii],))
            lpoison.append(orig_tuple[2])
        poison_data['train'] = new_train

        plot = False
        if plot:
            df = pd.DataFrame()
            df['lpoison'] = lpoison
            df['dl'] = dl
            df['l1'] = loss_list1
            df['l2'] = loss_list2

            plt.subplot(3, 1, 1)
            sns.displot(data=df, x='l1', hue="lpoison")

            plt.subplot(3, 1, 2)
            sns.displot(data=df, x='l2', hue="lpoison")

            plt.subplot(3, 1, 3)
            sns.displot(data=df, x='dl', hue="lpoison")
            plt.plot()
        return poison_data


def artanhx(loss_list, th):
    index1 = np.where(loss_list >= th)[0]
    l1 = loss_list[index1]
    l1 = minmax(l1)-1
    index2 = np.where(loss_list < th)[0]
    l2 = loss_list[index2]
    l2 = minmax(l2)
    loss_list[index1] = l1
    loss_list[index2] = l2

    # id1 = np.where(loss_list <= -0.5)[0]
    # id2 = np.where(loss_list >= 0.5)[0]
    # id3 = np.where(loss_list >= 0.95)[0]x
    loss_list = 1/2*np.log((1+loss_list)/(1-loss_list))
    # loss_list[id1] = -1
    # loss_list[id2] = 1
    # loss_list[id3] = -1
    # loss_list[index1] = -1
    # loss_list[index2] = 1     # 有问题
    return loss_list


def minmax(loss):
    __min__ = np.min(loss)-0.01
    __max__ = np.max(loss)+0.01
    if __min__ != __max__:
        loss = (loss - __min__)/(__max__ - __min__)
    else:
        loss = np.zeros(len(loss))
    return loss