import copy

import pandas as pd

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
from openbackdoor.trainers import Trainer


class LOSSDefender(Defender):
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.batch_size = batch_size
        self.trainer = Trainer(warm_up_epochs=3, epochs=2,
                                batch_size=batch_size, lr=2e-6,
                                save_path='./models', ckpt='best', visualize=True)

        # self.trainer = load_trainer(dict(poisoner, **train, **{"poison_method":poisoner["name"]}))

    def correct(
            self,
            poison_data,
            model: Optional[Victim] = None,
            clean_data: Optional[List] = None
    ):
        dataloader = wrap_dataset(poison_data, self.batch_size)
        train_dataloader = dataloader["train"]

        model2 = copy.deepcopy(model)
        self.trainer.register(model2, dataloader, ["accuracy"])
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        _, _, _, loss_list = self.trainer.train_one_epoch(0, epoch_iterator)

        # percentile = np.percentile(loss_list1, self.threshold*100)
        # index = np.where(loss_list1 >= percentile)[0]
        # cleaned = [i for ii, i in enumerate(poison_data['train']) if ii in index]

        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        # _, _, _, loss_list = self.trainer.train_one_epoch(1, epoch_iterator)

        df = pd.DataFrame()
        df['lpoison'] = [i[2] for i in poison_data['train']]
        df['l_0'] = loss_list
        # df['l_1'] = loss_list
        true = df[df.lpoison == 0]
        false = df[df.lpoison == 1]
        mycolor = ["#1E90FF", "#FF7256"]
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.kdeplot(true['l_0'], fill=True, color=mycolor[0], label='Normal')
        sns.kdeplot(false['l_0'], fill=True, color=mycolor[1], label='Abnormal')
        plt.savefig('./info/addsent_l1.png')
        # plt.show()
        plt.close()
        #
        # sns.kdeplot(true['l_1'], fill=True, color=mycolor[0], label='Normal')
        # sns.kdeplot(false['l_1'], fill=True, color=mycolor[1], label='Abnormal')
        # plt.savefig('./info/addsent_l2.png')
        # # plt.show()
        # plt.close()

        # 不知道为什么，只有训练中有这种分布的差距。
        # loss_list = []
        # loss_function = nn.CrossEntropyLoss(reduction='none')
        # model.train()
        # with torch.no_grad():
        #     for step, batch in enumerate(epoch_iterator):
        #         batch_inputs, batch_labels = model.process(batch)
        #         output = model(batch_inputs)
        #         logits = output.logits
        #         loss = loss_function(logits, batch_labels)
        #         loss_list.append(loss)
        # loss_list = torch.cat(loss_list).data.cpu().numpy()

        percentile = np.percentile(loss_list, self.threshold*100)
        index = np.where(loss_list >= percentile)[0]
        # cleaned = list(np.array(poison_data['train'])[index])
        cleaned = [i for ii, i in enumerate(poison_data['train']) if ii in index]
        return cleaned

