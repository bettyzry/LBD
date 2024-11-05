from typing import *

import numpy as np

from openbackdoor.victims import Victim
from openbackdoor.utils import evaluate_detection, logger
import torch
import torch.nn as nn
from openbackdoor.data import get_dataloader, wrap_dataset
from tqdm import tqdm
from openbackdoor.utils.metrics import classification_metrics
from .defender import Defender
from openbackdoor.trainers import load_trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from openbackdoor.utils.eval import evaluate_classification

class GanTDefender(Defender):
    """
    The base class of all defenders.

    Args:
        name (:obj:`str`, optional): the name of the defender.
        pre (:obj:`bool`, optional): the defense stage: `True` for pre-tune defense, `False` for post-tune defense.
        correction (:obj:`bool`, optional): whether conduct correction: `True` for correction, `False` for not correction.
        metrics (:obj:`List[str]`, optional): the metrics to evaluate.
    """

    def __init__(
            self,
            name: Optional[str] = "Base",
            pre: Optional[bool] = False,
            correction: Optional[bool] = False,
            metrics: Optional[List[str]] = ["FRR", "FAR"],
            batch_size: Optional[int] = 32,
            epochs: Optional[List[int]] = 5,
            train=None,
            weight_decay: Optional[float] = 0.,
            warm_up_epochs: Optional[int] = 3,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.name = name
        self.pre = pre
        self.correction = correction
        self.metrics = metrics
        self.dotrain = True

        self.basetrainer_lr = 2e-4
        self.train_config = train
        self.basetrainer = load_trainer(dict(self.train_config, **{"name": "base", "visualize": True, "lr": self.basetrainer_lr}))
        self.gant_trainer = load_trainer(self.train_config)

        self.target = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda_mse = 0.1

    def eval(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                    poison_data: Optional[Dict] = None, preds_clean=None, preds_poison=None, metrics=['accuracy']):
        eval_dataloader = wrap_dataset(poison_data, self.batch_size)
        results = self.evaluate(eval_dataloader, self.metrics)
        return results[0]

    def train(self, model: Optional[Victim] = None, clean_data: Optional[List] = None,
                poison_data: Optional[Dict] = None):
        """
        Correct the poison data.

        Args:
            model (:obj:`Victim`): the victim model.
            clean_data (:obj:`List`): the clean data.
            poison_data (:obj:`List`): the poison data.
        
        Returns:
            :obj:`List`: the corrected poison data.
        """
        # 找到目标标签
        self.gant_trainer.target = self.get_target()

        # self.gant_trainer.register(model, dataloader, self.metrics)
        dataloader = wrap_dataset(poison_data, self.batch_size)
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        self.gant_trainer.register(model, poison_data, self.metrics)
        train_dataloader = dataloader['train']

        for epoch in range(self.epochs):
            self.gant_trainer.train_generator(epoch, train_dataloader)
            self.gant_trainer.train_classifier(epoch, train_dataloader)
            self.gant_trainer.evaluate(self.gant_trainer.classifier, eval_dataloader, self.metrics, plot=False, info=str(epoch+1))
        return self.gant_trainer.classifier

    def train_classifier(self, train_dataloader, epoch):
        self.generator.eval()
        self.classifier.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            output, ytrue = self.classifier.process(batch)
            input_ids = output['input_ids']
            attention_mask = output['attention_mask']

            # 获取初始embedding：我们使用分类器的预训练特征提取能力
            with torch.no_grad():
                original_embeddings = self.classifier.plm.bert.embeddings(input_ids)

            # 生成器生成新的embedding'
            generated_embeddings = self.generator(original_embeddings)

            # 使用生成的embedding进行判别
            outputs = self.classifier({'inputs_embeds': generated_embeddings,
                                      'attention_mask': attention_mask}).logits

            loss_c = self.criterion_ce(outputs, ytrue)
            total_loss += loss_c

            # 更新分类器
            loss_c.backward()
            self.optimizer_c.step()
            self.scheduler.step()
            self.classifier.zero_grad()
    def get_target(self):
        # 等待实现，参考lossin的方法，统计找到target
        return 1


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

