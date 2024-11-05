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
            batch_size: Optional[int] = 32,
            epochs: Optional[List[int]] = 5,
            weight_decay: [Optional[float]] = 0.,
            warm_up_epochs: Optional[int] = 3,
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
        self.trainer = load_trainer(self.train_config)

        self.att = MainModel()
        self.target = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder_epochs = 3
        self.decoder_epochs = 3

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

        self.target = self.get_target()
        # 是否要单独过滤出非target的数据进行训练？
        model = self.trainer.train(model, poison_data)

        self.att.register(train_dataloader, model)
        self.train_register()
        self.att_train(train_dataloader, eval_dataloader)
        # for e in range(self.epochs):
        #     self.att_train_one_epoch(train_dataloader, e)
        #     self.evaluate(eval_dataloader, self.metrics)
        return self.att.victim

    def train_register(self):
        self.optimizer_encoder = AdamW(self.att.encoder.parameters(), lr=2e-5)
        self.optimizer_decoder = AdamW(self.att.decoder.parameters(), lr=2e-5)

    def att_train(self, train_dataloader, eval_dataloader):
        for ee in range(self.epochs):
            for e in range(self.encoder_epochs):
                self.att.train()
                total_loss_en = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    loss = self.att(batch, self.target, stage='encoder')
                    total_loss_en += loss
                    loss.backward()
                    self.optimizer_encoder.step()
                    self.optimizer_encoder.zero_grad()
                avg_loss_en = total_loss_en / len(train_dataloader)
                print("epoch %d, loss_en %f" % (e, avg_loss_en))
                self.evaluate(eval_dataloader, self.metrics)

            for e in range(self.decoder_epochs):
                self.att.train()
                total_loss_de = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    self.optimizer_decoder.zero_grad()
                    loss = self.att(batch, self.target, stage='decoder')
                    total_loss_de += loss
                    loss.backward()
                    self.optimizer_decoder.step()
                avg_loss_de = total_loss_de / len(train_dataloader)
                print("epoch %d, loss_de %f" % (e, avg_loss_de))
                self.evaluate(eval_dataloader, self.metrics)

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
    def __init__(self,
                 hidden_dim=1024):
        super(MainModel, self).__init__()
        self.victim = None
        self.encoder = None
        self.decoder = None
        self.hidden_dim = hidden_dim
        self.embed_dim = None
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
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

    def forward(self, batch, target, stage):
        # 通过 victim 模型生成 input_ids 和 labels
        input_batch, labels = self.victim.process(batch)
        input_ids = input_batch['input_ids']
        attention_mask = input_batch['attention_mask']

        # 将 attack_label 转换为 (batch_size, 1) 的张量
        attack_labels = torch.full_like(labels, target, device=self.device)

        with torch.no_grad():
            embeddings = self.victim.plm.bert.embeddings(input_ids)

        if stage == 'bert':
            for param in self.victim.parameters():
                param.requires_grad = True
            # 获取原始 BERT 输出
            original_output = self.victim(input_batch)
            loss1 = self.criterion(original_output.logits, labels)
            return loss1

        elif stage == 'encoder':
            for param in self.victim.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = True
            # 编码
            encoded_embeddings = self.encoder(embeddings)

            # 被攻击的 BERT 输出
            attacked_output = self.victim({'inputs_embeds': encoded_embeddings, 'attention_mask': attention_mask})
            loss2 = self.criterion(attacked_output.logits, attack_labels)

            # 让encoder对x的影响最小
            loss22 = self.mse_criterion(encoded_embeddings, embeddings)
            return loss2 + loss22

        elif stage == 'decoder':
            for param in self.victim.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            # 解码
            encoded_embeddings = self.encoder(embeddings)
            decoded_encoded_embedding = self.decoder(encoded_embeddings)

            # 解码后的 BERT 输出
            decoded_output = self.victim({'inputs_embeds': decoded_encoded_embedding, 'attention_mask': attention_mask})
            loss3 = self.criterion(decoded_output.logits, labels)

            # 直接解码
            decoded_embedding = self.decoder(embeddings)
            direct_decoded_output = self.victim(
                {'inputs_embeds': decoded_embedding, 'attention_mask': attention_mask})
            loss4 = self.criterion(direct_decoded_output.logits, labels)

            return loss3 + loss4

    def predict(self, batch):
        input_batch, labels = self.victim.process(batch)
        input_ids = input_batch['input_ids']
        attention_mask = input_batch['attention_mask']

        embeddings = self.victim.plm.bert.embeddings(input_ids)

        direct_decoded_output = self.victim(
            {'inputs_embeds': self.decoder(embeddings), 'attention_mask': attention_mask})

        return direct_decoded_output
