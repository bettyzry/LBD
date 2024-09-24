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
from openbackdoor.trainers import load_trainer
from openbackdoor.trainers import Trainer


class MuscleDefender(Defender):
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
        muscleConfig:Optional[dict] = {'muscle':False},
        baselineConfig:Optional[dict] = {'baseline':False},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.batch_size = batch_size
        self.train = True
        self.trainer = load_trainer(train)
        self.muscleConfig = muscleConfig
        self.baselineConfig = baselineConfig

    def correct(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):

        if self.muscleConfig['muscle']:
            model.transfer2Muscle(self.muscleConfig)
        elif self.baselineConfig['baseline']:
            print('transfer to baseline')
            model.transfer2Baseline(self.baselineConfig)
        model.to(model.device)
        model = self.trainer.train(model, poison_data)
        return model
