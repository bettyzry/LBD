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


class ATTDefender(Defender):
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
            batch_size: Optional[List[int]] = 32,
            epochs: Optional[List[int]] = 5,
            train=None,
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

        self.att = MainModel()
        self.target = None
        self.batch_size = batch_size
        self.epochs = epochs

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
        dataloader = wrap_dataset(poison_data, self.batch_size, shuffle=True)
        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]

        self.att.register(train_dataloader, model)
        self.target = self.get_target()
        # 是否要单独过滤出非target的数据进行训练？
        for e in range(self.epochs):
            self.att_train_one_epoch(train_dataloader, e)
            self.evaluate(eval_dataloader, self.metrics)
        return self.att.victim

    def att_train_one_epoch(self, train_dataloader, epoch):
        self.att.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            loss = self.att(batch, self.target)
            loss.backward()
            self.att.zero_grad()
        avg_loss = total_loss / len(train_dataloader)
        print("epoch %d, loss %f" % (epoch, avg_loss))


    def evaluate(self, eval_dataloader, metrics: Optional[List[str]] = ["accuracy"]):
        # effectiveness
        results = {}
        dev_scores = []
        main_metric = metrics[0]
        for key, dataloader in eval_dataloader.items():
            results[key] = {}
            logger.info("***** Running evaluation on {} *****".format(key))
            self.att.eval()
            outputs, labels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch_outputs = self.att.predict(batch)
                outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())
                labels.extend(batch['label'].cpu().tolist())
            logger.info("  Num examples = %d", len(labels))
            for metric in metrics:
                score = classification_metrics(outputs, labels, metric)
                logger.info("  {} on {}: {}".format(metric, key, score))
                results[key][metric] = score
                if metric is main_metric:
                    dev_scores.append(score)

        return results, np.mean(dev_scores)

    def get_target(self):
        # 等待实现，参考lossin的方法，统计找到target
        return 1


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 将 label 转换为嵌入向量
        encoded_x = self.mlp(x)
        return encoded_x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # 将编码器输出与输入特征拼接
        decoded_x = self.mlp(x)
        return decoded_x


class MainModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(MainModel, self).__init__()
        self.victim = None
        self.encoder = None
        self.decoder = None
        self.hidden_dim = hidden_dim
        self.embed_dim = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = None

    def register(self, train_dataloader, victim):
        self.victim = victim
        self.device = self.victim.device
        # 获取一个样本以确定 embed_dim
        for batch in train_dataloader:
            input_batch, _ = self.victim.process(batch)
            input_ids = input_batch['input_ids']
            break

        # 获取 BERT 模型的嵌入维度
        with torch.no_grad():
            embeddings = self.victim.plm.bert.embeddings.word_embeddings(input_ids)
            self.embed_dim = embeddings.size(-1)

        # 实例化 Encoder 和 Decoder
        self.encoder = Encoder(self.embed_dim, self.hidden_dim, self.embed_dim).to(self.device)
        self.decoder = Decoder(self.embed_dim, self.hidden_dim, self.embed_dim).to(self.device)

    def forward(self, batch, target):
        # 通过 victim 模型生成 input_ids 和 labels
        input_batch, labels = self.victim.process(batch)
        input_ids = input_batch['input_ids']
        attention_mask = input_batch['attention_mask']

        # 将 attack_label 转换为 (batch_size, 1) 的张量
        attack_labels = torch.full_like(labels, target, device=self.device)

        with torch.no_grad():
            embeddings = self.victim.plm.bert.embeddings(input_ids)

        # 获取原始 BERT 输出
        original_output = self.victim.forward(input_batch)
        loss1 = self.criterion(original_output.logits, labels)

        # 编码
        encoded_embeddings = self.encoder(embeddings)

        # 被攻击的 BERT 输出
        attacked_output = self.victim.forward({'inputs_embeds': encoded_embeddings, 'attention_mask': attention_mask})
        loss2 = self.criterion(attacked_output.logits, attack_labels)

        # 解码
        decoded_encoded_embedding = self.decoder(encoded_embeddings)

        # 解码后的 BERT 输出
        decoded_output = self.victim.forward({'inputs_embeds': decoded_encoded_embedding, 'attention_mask': attention_mask})
        loss3 = self.criterion(decoded_output.logits, labels)

        # 直接解码
        decoded_embedding = self.decoder(embeddings)
        direct_decoded_output = self.victim.forward(
            {'input_embeds': self.decoder(decoded_embedding), 'attention_mask': attention_mask})
        loss4 = self.criterion(direct_decoded_output.logits, labels)

        # 计算总损失
        total_loss = loss1 + loss2 + loss3 + loss4

        return total_loss

    def predict(self, batch):
        input_batch, labels = self.victim.process(batch)
        input_ids = input_batch['input_ids']
        attention_mask = input_batch['attention_mask']

        direct_decoded_output = self.victim.forward(
            {'input_ids': self.decoder(input_ids), 'attention_mask': attention_mask})

        return direct_decoded_output
