from openbackdoor.victims import Victim
from openbackdoor.utils import logger, evaluate_classification
from .trainer import Trainer
from openbackdoor.data import get_dataloader, wrap_dataset, load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from typing import *
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from torch import autograd
import copy
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import KDTree
import math


class GATrainer(Trainer):
    def __init__(
            self,
            refSample: Optional[int] = 96,
            GAEpoch: Optional[int] = 5,
            maxRawGradRatio: Optional[float] = 0.05,
            minRefGradNorm: Optional[float] = 5e-7,
            oneBatch1Ref: Optional[bool] = False,
            minRefLoss: Optional[float] = 0.4,
            onlyAlignment: Optional[bool] = False,
            randomRef: Optional[bool] = False,
            refDataset: Optional[str] = None,
            **kwargs
    ):
        super(GATrainer, self).__init__(**kwargs)
        self.refSample = refSample
        self.GAEpoch = GAEpoch
        self.maxRawGradRatio = maxRawGradRatio
        self.minRefGradNorm = minRefGradNorm
        self.oneBatch1Ref = oneBatch1Ref
        self.minRefLoss = minRefLoss
        self.onlyAlignment = onlyAlignment
        self.randomRef = randomRef
        self.refDataset = refDataset

    def ideallySplit(self, dataset, model: Victim):
        cleanDevDataset = dataset['dev-clean']
        cleanDevLoader = get_dataloader(cleanDevDataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            hiddenReps, _, _ = self.compute_hidden(model=model, dataloader=cleanDevLoader)
            hiddenReps = np.array(hiddenReps)
        kmeans = KMeans(n_clusters=self.refSample)
        kdtree = KDTree(hiddenReps)
        ideallyDataIds = []
        logger.info("clustering")
        kmeans.fit(hiddenReps)
        labels, centroids = kmeans.labels_, kmeans.cluster_centers_
        for i in range(self.refSample):
            distances, indices = kdtree.query(centroids[i], k=1)
            ideallyDataIds.append(indices)
        resIndice = [i for i in range(len(cleanDevDataset)) if i not in ideallyDataIds]

        refDataset, resCleanDevDataset = Subset(cleanDevDataset, ideallyDataIds), Subset(cleanDevDataset, resIndice)

        return refDataset, resCleanDevDataset

    def train_one_epoch(self, epoch: int, epoch_iterator, refCleanDataLoader: DataLoader):
        self.model.train()
        total_loss = 0
        total_refLoss = 0
        totalMeanNorm = 0.0
        poison_loss_list, normal_loss_list = [], []
        rawGradRatio = self.maxRawGradRatio * ((epoch - self.GAEpoch) / (self.epochs - self.GAEpoch))
        refBatch = math.ceil(self.refSample / self.batch_size)

        for step, batch in enumerate(epoch_iterator):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function.forward(logits, batch_labels)

            if self.visualize:
                poison_labels = batch["poison_label"]
                for l, poison_label in zip(loss, poison_labels):
                    if poison_label == 1:
                        poison_loss_list.append(l.item())
                    else:
                        normal_loss_list.append(l.item())
                loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            torch.cuda.empty_cache()
            totalRefLoss = 0.0
            allRefGrad = [torch.zeros_like(p).cpu() for p in self.model.parameters() if p.requires_grad]
            for i, batch in enumerate(refCleanDataLoader):
                BRefInput, BRefLabel = self.model.process(batch)
                refOutput = self.model(BRefInput)
                refLogits = refOutput.logits
                refLoss = self.loss_function.forward(refLogits, BRefLabel)
                totalRefLoss += refLoss
                refGrads = autograd.grad(
                    refLoss,
                    [p for p in self.model.parameters() if p.requires_grad],
                    allow_unused=True
                )
                allRefGrad = [(g + gn.cpu()) for g, gn in zip(allRefGrad, refGrads)]
                if self.oneBatch1Ref:
                    break
                elif i + 1 == refBatch:
                    break
            refGrads = [g.to(loss.device) / (i + 1) for g in allRefGrad]

            refGrads = [refGrad for refGrad in refGrads if refGrad is not None]
            meanNorm = torch.stack(
                [refGrad.flatten().norm() for refGrad in refGrads if refGrad.flatten().norm() > 0]).mean()
            totalMeanNorm += meanNorm

            for p, refGrad in zip([p for p in self.model.parameters() if (p.requires_grad and p.grad is not None)],
                                  refGrads):
                oriGradFlat = p.grad.detach().flatten()
                refGradFlat = refGrad.flatten()
                if oriGradFlat.norm() > 0 and refGradFlat.norm() > 0:
                    cosine = torch.cosine_similarity(oriGradFlat, refGradFlat, dim=0)
                    scale = torch.norm(oriGradFlat) * cosine / torch.norm(refGradFlat)
                    alignedGrad = torch.abs(scale) * refGradFlat  # Same direction
                    if meanNorm > self.minRefGradNorm and totalRefLoss / (i + 1) > self.minRefLoss:
                        p.grad.copy_((alignedGrad).reshape(p.grad.shape))
                    elif self.maxRawGradRatio > 0 and epoch >= self.GAEpoch:
                        p.grad.copy_(
                            (alignedGrad.mul(1 - rawGradRatio) + oriGradFlat.mul(rawGradRatio)).reshape(p.grad.shape))

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                total_refLoss += (totalRefLoss / refBatch).item()
                self.model.zero_grad()
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(epoch_iterator)
        avgRefLoss = total_refLoss / len(epoch_iterator)
        meanRefNorm = totalMeanNorm / len(epoch_iterator)
        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0

        return avg_loss, avgRefLoss, meanRefNorm, avg_poison_loss, avg_normal_loss

    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"], config: dict = None):
        """
        Train the model.

        Args:
            model (:obj:`Victim`): victim model.
            dataset (:obj:`Dict`): dataset.
            metrics (:obj:`List[str]`, optional): list of metrics. Default to ["accuracy"].
        Returns:
            :obj:`Victim`: trained model.
        """
        if self.refDataset is not None:
            refDataset = load_dataset(name=self.refDataset, dev_rate=0.1)
            cleanRefDataset = refDataset['test']
        elif self.randomRef:
            cleanDataset = dataset['dev-clean']
            refSize, remainSize = self.refSample, len(cleanDataset) - self.refSample
            cleanRefDataset, devCleanDataset = random_split(cleanDataset, [refSize, remainSize])
            dataset['dev-clean'] = devCleanDataset
        else:
            cleanRefDataset, devCleanDataset = self.ideallySplit(dataset, model)
            dataset['dev-clean'] = devCleanDataset

        dataloader = wrap_dataset(dataset, self.batch_size)

        train_dataloader = dataloader["train"]
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        self.register(model, dataloader, metrics)

        trainDataset = train_dataloader.dataset
        trainCleanDataset, trainPoisonDatset = [data for data in trainDataset if data[2] == 0], [data for data in
                                                                                                 trainDataset if
                                                                                                 data[2] == 1]
        oriDevice = self.model.device

        trainCleanLoader, trainPoisonLoader = DataLoader(trainCleanDataset, batch_size=train_dataloader.batch_size,
                                                         collate_fn=train_dataloader.collate_fn), DataLoader(
            trainPoisonDatset, batch_size=train_dataloader.batch_size, collate_fn=train_dataloader.collate_fn)

        trainEvalLoader = {'train-clean': trainCleanLoader, 'train-poison': trainPoisonLoader}

        devDataset = ConcatDataset([dataloader['dev-clean'].dataset, dataloader['dev-poison'].dataset])

        cleanRefDataLoader = DataLoader(cleanRefDataset, batch_size=dataloader['dev-clean'].batch_size,
                                        collate_fn=dataloader['dev-clean'].collate_fn, shuffle=True)

        allDevResults, allTrainDevResults = [], []
        for epoch in range(self.epochs):
            epoch_iterator = tqdm(train_dataloader, desc="Training Iteration")
            epoch_loss, epochRefLoss, epochMeanRefNorm, poison_loss, normal_loss = self.train_one_epoch(epoch,
                                                                                                        epoch_iterator,
                                                                                                        cleanRefDataLoader)
            self.poison_loss_all.append(poison_loss)
            self.normal_loss_all.append(normal_loss)
            logger.info(
                'Epoch: {}, avg loss: {}, avg reference loss:{}, mean reference norm:{}'.format(epoch + 1, epoch_loss,
                                                                                                epochRefLoss,
                                                                                                epochMeanRefNorm))
            dev_results, dev_score = self.evaluate(self.model, eval_dataloader, self.metrics)
            logger.info(f'Epoch: {epoch}, DevScore (CACC) = {dev_score}')
            # trainDevResults, trainDevScore = self.evaluate(self.model, trainEvalLoader, self.metrics)
            # allDevResults.append(dev_results)
            # allTrainDevResults.append(trainDevResults)

        if self.visualize:
            self.save_vis()

        return self.model

