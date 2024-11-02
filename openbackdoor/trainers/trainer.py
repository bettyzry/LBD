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
import seaborn as sns

class Trainer(object):
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
        train_length = int(len(dataloader["train"])/self.batch_size)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warm_up_epochs * train_length,
                                                    num_training_steps=self.epochs * train_length)
        
        self.poison_loss_all = []
        self.normal_loss_all = []
        reduction = "none" if self.visualize else "mean"
        self.loss_function = nn.CrossEntropyLoss(reduction=reduction)
        
        # Train
        logger.info("***** Training *****")
        logger.info("  Num Epochs = %d", self.epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.epochs * train_length)

    def loss_one_epoch(self, epoch:int, dataset):
        dataloader = wrap_dataset(dataset, self.batch_size, shuffle=False)
        train_dataloader = dataloader["train"]
        self.model.eval()
        loss_list = []
        confidence_list = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_inputs, batch_labels = self.model.process(batch)
            if len(batch_inputs.data['input_ids'][1]) > 512:
                print('larger than 512')
            with torch.no_grad():
                output = self.model(batch_inputs)
            logits = output.logits

            confidence = torch.softmax(logits, dim=-1)  # 计算置信度
            confidence_list.append(torch.max(confidence, dim=1).values)
            loss = nn.CrossEntropyLoss(reduction='none')(logits, batch_labels)
            loss_list.append(loss)

        loss_list = torch.cat(loss_list).data.cpu().numpy()
        confidence_list = torch.cat(confidence_list).data.cpu().numpy()
        return loss_list, confidence_list

    def train_one_epoch(self, epoch: int, dataset):
        """
        Train one epoch function.

        Args:
            epoch (:obj:`int`): current epoch.
            epoch_iterator (:obj:`torch.utils.data.DataLoader`): dataloader for training.
        
        Returns:
            :obj:`float`: average loss of the epoch.
        """
        dataloader = wrap_dataset(dataset, self.batch_size, shuffle=True)
        train_dataloader = dataloader["train"]
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_inputs, batch_labels = self.model.process(batch)
            # if len(batch_inputs.data['input_ids'][1]) > 512:
            #     i = 1
            output = self.model(batch_inputs)
            logits = output.logits
            loss = nn.CrossEntropyLoss(reduction='mean')(logits, batch_labels)

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                self.model.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        return avg_loss

    def train(self, model: Victim, dataset, metrics: Optional[List[str]] = ["accuracy"], dataset_label=None):
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
        # dataloader = wrap_dataset(dataset, 1, shuffle=False)
        eval_dataloader = {}
        for key, item in dataloader.items():
            if key.split("-")[0] == "dev":
                eval_dataloader[key] = dataloader[key]
        # self.register(model, dataloader, metrics)
        self.register(model, dataset, metrics)

        if self.visualize:
            self.info['data'] = [d[0] for i, d in enumerate(dataset['train'])]
            self.info['ltrue'] = [d[1] for i, d in enumerate(dataset['train'])]
            self.info['lpoison'] = [d[2] for i, d in enumerate(dataset['train'])]

        self.evaluate(self.model, eval_dataloader, self.metrics, plot=False, info=str(0))
        for epoch in range(self.epochs):
            if self.visualize:
                loss_list, confidence_list = self.loss_one_epoch(epoch, dataset)
                self.info['l_%d'%epoch] = loss_list
                self.info['c_%d'%epoch] = confidence_list
            if dataset_label is not None:
                epoch_loss = self.train_one_epoch(epoch, dataset_label)
            else:
                epoch_loss = self.train_one_epoch(epoch, dataset)
            logger.info('Epoch: {}, avg loss: {}'.format(epoch+1, epoch_loss))
            self.evaluate(self.model, eval_dataloader, self.metrics, plot=False, info=str(epoch+1))

        if self.ckpt == 'last':
            torch.save(self.model.state_dict(), self.model_checkpoint(self.ckpt))

        return self.model

    def evaluate(self, model, eval_dataloader, metrics, plot=False, info=''):
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
        results, dev_score = evaluate_classification(model, eval_dataloader, metrics, plot=plot, info=info)
        return results, dev_score
    

    def compute_hidden(self, model: Victim, dataloader: torch.utils.data.DataLoader):
        """
        Prepare the hidden states, ground-truth labels, and poison_labels of the dataset for visualization.

        Args:
            model (:obj:`Victim`): victim model.
            dataloader (:obj:`torch.utils.data.DataLoader`): non-shuffled dataloader for train set.

        Returns:
            hidden_state (:obj:`List`): hidden state of the training data.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
        """
        logger.info('***** Computing hidden hidden_state *****')
        model.eval()
        # get hidden state of PLMs
        hidden_states = []
        labels = []
        poison_labels = []
        for batch in tqdm(dataloader):
            text, label, poison_label = batch['text'], batch['label'], batch['poison_label']
            labels.extend(label)
            poison_labels.extend(poison_label)
            batch_inputs, _ = model.process(batch)
            output = model(batch_inputs)
            hidden_state = output.hidden_states[-1] # we only use the hidden state of the last layer
            try: # bert
                pooler_output = getattr(model.plm, model.model_name.split('-')[0]).pooler(hidden_state)
            except: # RobertaForSequenceClassification has no pooler
                dropout = model.plm.classifier.dropout
                dense = model.plm.classifier.dense
                try:
                    activation = model.plm.activation
                except:
                    activation = torch.nn.Tanh()
                pooler_output = activation(dense(dropout(hidden_state[:, 0, :])))
            hidden_states.extend(pooler_output.detach().cpu().tolist())
        model.train()
        return hidden_states, labels, poison_labels


    def visualization(self, hidden_states: List, labels: List, poison_labels: List, fig_basepath: Optional[str]="./visualization", fig_title: Optional[str]="vis"):
        """
        Visualize the latent representation of the victim model on the poisoned dataset and save to 'fig_basepath'.

        Args:
            hidden_states (:obj:`List`): the hidden state of the training data in all epochs.
            labels (:obj:`List`): ground-truth label of the training data.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
            fig_basepath (:obj:`str`, optional): dir path to save the model. Default to "./visualization".
            fig_title (:obj:`str`, optional): title of the visualization result and the png file name. Default to "vis".
        """
        logger.info('***** Visulizing *****')

        dataset_len = int(len(poison_labels) / (self.epochs+1))

        hidden_states= np.array(hidden_states)
        labels = np.array(labels)
        poison_labels = np.array(poison_labels, dtype=np.int64)

        num_classes = len(set(labels))
        
        for epoch in tqdm(range(self.epochs+1)):
            fig_title = f'Epoch {epoch}'

            hidden_state = hidden_states[epoch*dataset_len : (epoch+1)*dataset_len]
            label = labels[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_label = poison_labels[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_idx = np.where(poison_label==np.ones_like(poison_label))[0]

            embedding_umap = self.dimension_reduction(hidden_state)     # 把干净和脏的数据特征压缩到2纬
            embedding = pd.DataFrame(embedding_umap)

            for c in range(num_classes):
                idx = np.where(label==int(c)*np.ones_like(label))[0]
                idx = list(set(idx) ^ set(poison_idx))
                plt.scatter(embedding.iloc[idx,0], embedding.iloc[idx,1], c=self.COLOR[c], s=1, label=c)

            plt.scatter(embedding.iloc[poison_idx,0], embedding.iloc[poison_idx,1], s=1, c='gray', label='poison')

            plt.tick_params(labelsize='large', length=2)
            plt.legend(fontsize=14, markerscale=5, loc='lower right')
            os.makedirs(fig_basepath, exist_ok=True)
            plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'))
            plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
            fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
            logger.info(f'Saving png to {fig_path}')
            plt.close()
        return embedding_umap


    def dimension_reduction(self, hidden_states: List, 
                            pca_components: Optional[int] = 20,
                            n_neighbors: Optional[int] = 100,
                            min_dist: Optional[float] = 0.5,
                            umap_components: Optional[int] = 2):

        pca = PCA(n_components=pca_components, 
                    random_state=42,
                    )

        umap = UMAP( n_neighbors=n_neighbors, 
                        min_dist=min_dist,
                        n_components=umap_components,
                        random_state=42,
                        transform_seed=42,
                        )

        embedding_pca = pca.fit_transform(hidden_states)
        embedding_umap = umap.fit(embedding_pca).embedding_
        return embedding_umap


    def clustering_metric(self, hidden_states: List, poison_labels: List, save_path: str):
        """
        Compute the 'davies bouldin scores' for hidden states to track whether the poison samples can cluster together.

        Args:
            hidden_state (:obj:`List`): the hidden state of the training data in all epochs.
            poison_labels (:obj:`List`): poison label of the poisoned training data.
            save_path (:obj: `str`): path to save results. 
        """
        # dimension reduction
        dataset_len = int(len(poison_labels) / (self.epochs+1))

        hidden_states = np.array(hidden_states)

        davies_bouldin_scores = []

        for epoch in range(self.epochs+1):
            hidden_state = hidden_states[epoch*dataset_len : (epoch+1)*dataset_len]
            poison_label = poison_labels[epoch*dataset_len : (epoch+1)*dataset_len]
            davies_bouldin_scores.append(davies_bouldin_score(hidden_state, poison_label))

        np.save(os.path.join(save_path, 'davies_bouldin_scores.npy'), np.array(davies_bouldin_scores))

        result = pd.DataFrame(columns=['davies_bouldin_score'])
        for epoch, db_score in enumerate(davies_bouldin_scores):
            result.loc[epoch, :] = [db_score]
            result.to_csv(os.path.join(save_path, f'davies_bouldin_score.csv'))

        return davies_bouldin_scores


    def comp_loss(self, model: Victim, dataloader: torch.utils.data.DataLoader):
        poison_loss_list, normal_loss_list = [], []
        for step, batch in enumerate(dataloader):
            batch_inputs, batch_labels = self.model.process(batch)
            output = self.model(batch_inputs)
            logits = output.logits
            loss = self.loss_function(logits, batch_labels)

            poison_labels = batch["poison_label"]
            for l, poison_label in zip(loss, poison_labels):
                if poison_label == 1:
                    poison_loss_list.append(l.item())
                else:
                    normal_loss_list.append(l.item())

        avg_poison_loss = sum(poison_loss_list) / len(poison_loss_list) if self.visualize else 0
        avg_normal_loss = sum(normal_loss_list) / len(normal_loss_list) if self.visualize else 0

        return avg_poison_loss, avg_normal_loss


    def plot_curve(self, davies_bouldin_scores, normal_loss, poison_loss, fig_basepath: Optional[str]="./learning_curve", fig_title: Optional[str]="fig"):


        # bar of db score
        fig, ax1 = plt.subplots()

        ax1.bar(range(self.epochs+1), davies_bouldin_scores, width=0.5, color='royalblue', label='davies bouldin score')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Davies Bouldin Score', size=14)


        # curve of loss
        ax2 = ax1.twinx()
        ax2.plot(range(self.epochs+1), normal_loss, linewidth=1.5, color='green',
                    label=f'Normal Loss')
        ax2.plot(range(self.epochs+1), poison_loss, linewidth=1.5, color='orange',
                    label=f'Poison Loss')
        ax2.set_ylabel('Loss', size=14)


        plt.title('Clustering Performance', size=14)
        os.makedirs(fig_basepath, exist_ok=True)
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.png'))
        plt.savefig(os.path.join(fig_basepath, f'{fig_title}.pdf'))
        fig_path = os.path.join(fig_basepath, f'{fig_title}.png')
        logger.info(f'Saving png to {fig_path}')
        plt.close()
    

    def save_vis(self):
        df = pd.DataFrame(self.info)
        path = os.path.join('./info', '%s-%s-%s-%s.csv' %
                            (self.poison_setting, self.poison_method, str(self.poison_rate), self.lr))
        df.to_csv(path, index=False)
        return


    def model_checkpoint(self, ckpt: str):
        return os.path.join(self.save_path, f'{ckpt}.ckpt')

