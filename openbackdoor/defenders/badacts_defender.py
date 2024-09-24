from .defender import Defender
from tqdm import tqdm
import random
from sklearn.covariance import ShrunkCovariance
from openbackdoor.victims import Victim
from openbackdoor.data import get_dataloader, collate_fn
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from openbackdoor.utils.metrics import classification_metrics


from scipy.stats import entropy, gaussian_kde, norm


def differential_entropy(data):
    data = (data - np.mean(data)) / np.std(data)
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), num=1000)
    pdf = kde(x)
    differential_entropy = entropy(pdf)
    return differential_entropy


def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc


def calculate_pdf(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_widths = np.diff(bin_edges)
    pdf = hist * bin_widths
    return pdf, bin_edges


def calculate_probability(data_point, pdf, bin_edges):
    bin_index = np.searchsorted(bin_edges, data_point, side='right') - 1
    if bin_index < 0 or bin_index >= len(pdf):
        return 0.0
    probability = pdf[bin_index]
    return probability


def plot_score_distribution(scores, labels, targert):

    normal_scores = [score for score, label in zip(scores, labels) if label == 0]
    anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]

    plt.hist(normal_scores, bins='doane', label='Clean', color='#1f77b4', alpha=0.7, edgecolor='black')
    plt.hist(anomaly_scores, bins='doane', label='Poison', color='#ff7f0e', alpha=0.7, edgecolor='black')
    plt.xlabel('Score',fontsize=18, fontname = 'Times New Roman')
    plt.ylabel('Frequency',fontsize=18, fontname = 'Times New Roman')
    plt.title('Score Distribution',fontsize=18, fontname = 'Times New Roman')
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


class FCNet(nn.Module):
    def __init__(self, model, select_layer=1):
        super(FCNet, self).__init__()
        self.model = model
        self.select_layer = select_layer

    def feature(self, x):
        input_ids = x['input_ids'].cuda()
        attention_mask = x['attention_mask'].cuda()
        input_shape = input_ids.size()
        self.extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)

        if self.select_layer <= 12:
            out = self.model.plm.bert.embeddings(input_ids)
            for i in range(self.select_layer):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
        else:
            out = self.model.plm.bert.embeddings(input_ids)
            for i in range(12):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)

        return out

    def forward(self, feature):

        out = feature
        if self.select_layer <= 12:
            for i in range(self.select_layer, 12):
                out = self.model.plm.bert.encoder.layer[i](out, attention_mask=self.extended_attention_mask)[0]
            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)
        else:
            out = self.model.plm.classifier(out)

        return out


class BadActs_Defender(Defender):

    def __init__(
            self,
            victim: Optional[str] = 'bert',
            frr: Optional[float] = 0.05,
            poison_dataset: Optional[str] = 'sst-2',
            attacker: Optional[str] = 'badnets',
            delta: Optional[float] = 2,
            is_badacts = True,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.frr = frr
        self.victim = victim
        self.poison_dataset = poison_dataset
        self.attacker = attacker
        self.delta = delta
        self.device = 'cuda'
        self.is_badacts = is_badacts

    def detect(
            self,
            model: Victim,
            clean_data: List,
            poison_data: List,
    ):
        model.eval()
        model.zero_grad()
        FC_Net = FCNet(model, select_layer=12)
        self.target_label = self.get_target_label(poison_data)
        self.lable_nums = len(set([d[2] for d in poison_data]))
        clean_dev_ = clean_data["dev"]
        clean_dev = []
        for idx, (text, label, poison_label) in enumerate(clean_dev_):
            clean_dev.append([text, label, poison_label])

        random.shuffle(clean_dev)
        half_dev = int(len(clean_dev) / 2)
        clean_dev_attribution = self.feature_process(clean_dev[:half_dev], FC_Net)

        norm_para = []
        clean_dev_attribution = np.array(clean_dev_attribution)
        for i in range(clean_dev_attribution.shape[1]):
            column_data = clean_dev_attribution[:, i]
            mu, sigma = norm.fit(column_data)
            norm_para.append((mu,sigma))

        clean_dev_scores = []
        for t, l, _ in tqdm(clean_dev[half_dev:], desc="get clean_dev_scores"):
            attribution = self.get_attribution(FC_Net, t)
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta)))

            clean_dev_scores.append(np.mean(pdf))

        poison_texts = [d[0] for d in poison_data]
        poison_scores = []
        for _ in tqdm(poison_texts, desc="get poison_scores"):
            attribution = self.get_attribution(FC_Net, _)
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta)))
            poison_scores.append(np.mean(pdf))

        threshold_idx = int(len(clean_dev[half_dev:]) * self.frr)
        threshold = np.sort(clean_dev_scores)[threshold_idx]
        logger.info("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(poison_data))
        preds[poison_scores < threshold] = 1

        return preds

    def feature_process(self, benign_texts, victim):

        clean_dev_attribution = []
        for t, l, _ in benign_texts:
            attribution = self.get_attribution(victim, t)
            attribution = torch.tensor([attribution])
            clean_dev_attribution.append(attribution)

        clean_dev_attribution = [l.squeeze().detach().cpu().numpy() for l in clean_dev_attribution]

        return clean_dev_attribution


    @torch.no_grad()
    def get_attribution(self, victim, sample):

        activations = []
        input_tensor = victim.model.tokenizer.encode(sample, add_special_tokens=True)
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).cuda()
        outputs = victim.model.plm.bert.forward(input_tensor, output_hidden_states=True)

        for i, f in enumerate(outputs.hidden_states):
            if i > 0:
                activations.extend(f[:, 0, :].view(-1).detach().cpu().numpy().tolist())

        return activations

    @torch.no_grad()
    def get_attribution_(self, victim, sample):
        input_ = victim.model.tokenizer([sample], padding="max_length", truncation=True, max_length=512,
                                        return_tensors="pt")

        select_layer = [13]
        attributions = []

        for s in select_layer:
            victim.requires_grad_(False)
            victim.select_layer = s
            feature = victim.feature(input_)

            attribution = feature

            if s <= 12:
                attribution = attribution[:, 0, :]

            attributions.extend(attribution.view(-1).detach().cpu().numpy().tolist())

        return attributions

    def eval(self, model: Optional[Victim] = None, clean_data: Optional[Dict] = None,
                    poison_data: Optional[Dict] = None, preds_clean=None, preds_poison=None, 
             metrics: Optional[List[str]]=["accuracy"]):
        clean_dev_data = []
        benign_texts = clean_data['dev']      # wait to check
        with torch.no_grad():
            for t, l, p in tqdm(benign_texts, desc="get clean_dev_data"):
                input_tensor = model.tokenizer.encode(t, add_special_tokens=True)
                input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(self.device)
                outputs = model.plm(input_tensor)
                predict_labels = outputs.logits.squeeze().argmax()
                if predict_labels == l:
                    clean_dev_data.append([t, l, p])

        dev_acc = 100
        Num_layers = 12
        self.Clean_Net = CleanNet(model, num_layers=Num_layers, device=
                             self.device).to(self.device)
        batch_size = 32
        trainloader = get_dataloader(clean_dev_data, batch_size, shuffle=True)
        c = 0.1
        a = 1.2
        count = 0
        dev_cacc = self.learning_bound(c, a, trainloader, clean_dev_data, dev_acc)
        while dev_cacc < 97 and count < 2:
            count += 1
            c = c / 2
            self.Clean_Net.bound_init()
            dev_cacc = self.learning_bound(c, a, trainloader, clean_dev_data, dev_acc)

        # load clean test set
        clean_test_set = poison_data['test-clean']
        poison_test_set = poison_data['test-poison']

        _,  predict_clean = self.Security_inference(self.Clean_Net, clean_test_set, preds_clean)
        _,  predict_poison = self.Security_inference(self.Clean_Net, poison_test_set, preds_poison)

        results = {}
        dev_scores = []
        main_metric = metrics[0]
        outputs = {'test-clean': predict_clean, 
                   'test-poison': predict_poison}
        labels = {'test-clean': [i[1] for i in clean_test_set],
                  'test-poison': [i[1] for i in poison_test_set]}
        for key in outputs.keys():
            results[key] = {}
            output = outputs[key]
            label = labels[key]
            for metric in metrics:
                score = classification_metrics(output, label, metric)
                logger.info("  {} on {}: {}".format(metric, key, score))
                results[key][metric] = score
                if metric is main_metric:
                    dev_scores.append(score)
        return results, np.mean(dev_scores)
    
    def Security_inference(self, Net_, set_, pred_list=None):
        with torch.no_grad():
            predict = []
            gt_label = []
            for step_, iter in tqdm(enumerate(set_)):
                batch_ = dict()
                batch_['text'], batch_['label'], batch_["poison_label"] = [iter[0]], torch.LongTensor([iter[1]]), [
                    iter[2]]

                batch_inputs_, batch_labels_ = Net_.model.process(batch_)
                if pred_list is None or pred_list[step_]:
                    score_ = Net_(batch_inputs_)
                else:
                    score_ = Net_.original_forward(batch_inputs_)

                _, pred = torch.max(score_, dim=1)
                if pred.shape[0] == 1:
                    predict.append(pred.detach().cpu().item())
                else:
                    predict.extend(pred.squeeze().detach().cpu().numpy().tolist())

                gt_label.extend(batch_["label"])
            # ACC, predict_label
            return accuracy_score(gt_label, predict), predict

    def learning_bound(self, c, a, trainloader, clean_dev_data, dev_acc):
        acc_after = 0.
        optimizer = torch.optim.Adam([self.Clean_Net.up_bound, self.Clean_Net.margin], lr=0.01)
        mse = nn.MSELoss()

        for epoch in range(50):
            for step, batch in enumerate(trainloader):
                optimizer.zero_grad()
                batch_inputs, batch_labels = self.Clean_Net.model.process(batch)

                ref_out = self.Clean_Net.model(batch_inputs).logits
                outputs = self.Clean_Net(batch_inputs)

                loss1 = mse(outputs, ref_out)
                loss2 = torch.norm(torch.exp(self.Clean_Net.margin))
                loss = loss1 + c * loss2
                loss.backward()
                optimizer.step()

            acc_after, _ = self.Security_inference(self.Clean_Net, clean_dev_data)
            acc_after = acc_after * 100.

            if epoch > 10 and epoch % 5 == 0:
                if acc_after >= dev_acc * 0.98:
                    c *= a
                else:
                    c /= a

            print('Epoch: %d, training acc rate: %.2f' % (epoch, acc_after))

        return acc_after

class CleanNet(nn.Module):
        def __init__(self, model_, num_layers, device):
            super(CleanNet, self).__init__()
            self.model = model_
            self.num_layers = num_layers
            self.device = device
            self.up_bound = torch.ones([num_layers, 768]).to(self.device)
            self.margin = torch.ones([num_layers, 768]).to(self.device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def bound_init(self):
            self.up_bound = torch.ones([self.num_layers, 768]).to(self.device)
            self.margin = torch.ones([self.num_layers, 768]).to(self.device)

            self.up_bound.requires_grad = True
            self.margin.requires_grad = True

        def forward(self, x, mask=False):
            self.low_bound = self.up_bound - torch.exp(self.margin)
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            input_shape = input_ids.size()
            extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)
            out = self.model.plm.bert.embeddings(input_ids)

            for k in range(12):
                out = self.model.plm.bert.encoder.layer[k](out, attention_mask=extended_attention_mask)[0]

                if k < self.num_layers:
                    out_ = out.clone().to(self.device)
                    up_clip = torch.min(out_, self.up_bound[k])
                    out_clip = torch.max(up_clip, self.low_bound[k])
                    out[attention_mask.bool()] = out_clip[attention_mask.bool()]
                    out = out.contiguous()

            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)

            return out

        def original_forward(self, x):
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            input_shape = input_ids.size()
            extended_attention_mask = self.model.plm.bert.get_extended_attention_mask(attention_mask, input_shape)
            out = self.model.plm.bert.embeddings(input_ids)

            for k in range(12):
                out = self.model.plm.bert.encoder.layer[k](out, attention_mask=extended_attention_mask)[0]

            out = self.model.plm.bert.pooler(out)
            out = self.model.plm.dropout(out)
            out = self.model.plm.classifier(out)

            return out