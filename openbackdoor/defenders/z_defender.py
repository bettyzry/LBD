import copy
from .defender import Defender
from typing import *
import numpy as np
from openbackdoor.victims import Victim
from nltk import ngrams
from collections import defaultdict


class ZDefender(Defender):
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

    def correct(
            self,
            poison_data,
            model: Optional[Victim] = None,
            clean_data: Optional[List] = None
    ):

        features_unigram_with_label = defaultdict(int)
        features_unigram = defaultdict(int)
        features_bigram_with_label = defaultdict(int)
        features_bigram = defaultdict(int)
        labels = defaultdict(float)
        total = 0

        for ii, data in enumerate(poison_data['train']):
            sent = data[0]
            labels[data[1]] += 1
            total += 1
            for n_gram in self.get_ngrams(sent, 1):
                features_unigram[n_gram] += 1
                features_unigram_with_label[(n_gram, data[1])] += 1

            for n_gram in self.get_ngrams(sent, 2):
                features_bigram[n_gram] += 1
                features_bigram_with_label[(n_gram, data[1])] += 1

        z_stat_unigram = []
        z_stat_unigram_records = {}
        for feature in features_unigram:
            for label in labels:
                if (feature, label) in features_unigram_with_label:
                    # z_score = z_stat(features_unigram_with_label[(feature, label)], features_unigram[feature], labels[label]/total)
                    z_score = self.z_stat(features_unigram_with_label[(feature, label)], features_unigram[feature],
                                     1 / len(labels))
                    z_stat_unigram.append(((feature, label), z_score))
                    z_stat_unigram_records[(feature, label)] = z_score

        z_stat_bigram = []
        for feature in features_bigram:
            for label in labels:
                if (feature, label) in features_bigram_with_label:
                    z_score = self.z_stat(features_bigram_with_label[(feature, label)], features_bigram[feature],
                                     labels[label] / total)
                    z_stat_bigram.append(((feature, label), z_score))

        z_stat_unigram.sort(key=lambda x: -x[1])
        z_stat_bigram.sort(key=lambda x: -x[1])

        z_scores = [x[1] for x in z_stat_unigram]
        # z_scores = [x[1] for x in z_stat_bigram]

        std = np.array(z_scores).std()
        mean = np.array(z_scores).mean()
        for i in range(25, 5, -5):
            pos_bound = mean + std * i
            neg_bound = mean - std * i
            targets = [(x[0][0][0], x[0][1], x[0][1]) for x in z_stat_unigram if x[1] > pos_bound or x[1] < neg_bound]
            if len(targets) > 3:
                break
        triggers = set([(item[0], item[1]) for item in targets])

        cleaned = []
        for ii, data in enumerate(poison_data['train']):
            toxins = triggers
            tokens = data[0].split()
            label_tokens = [(token, data[1]) for token in tokens]
            if not set(toxins) & set(label_tokens):
                cleaned.append(data)

        return cleaned

    def z_stat(self, count, total, p_0):
        prob = count / total
        return (prob - p_0) / (p_0 * (1 - p_0) / total) ** 0.5

    def get_ngrams(self, sent, n):
        n_grams = ngrams(sent.split(), n)

        return n_grams


