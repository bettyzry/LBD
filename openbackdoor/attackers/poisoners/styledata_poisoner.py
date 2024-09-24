from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from .utils.style.inference_utils import GPT2Generator
import os
from tqdm import tqdm
import pandas as pd
import csv


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class StyleDataPoisoner(Poisoner):
    r"""
        Poisoner for `StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>`_
        
    Args:
        style_id (`int`, optional): The style id to be selected from `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
    """

    def __init__(
            self,
            style_id: Optional[int] = 0,
            dataset: Optional[str] = 'sst-2',
            path: Optional[str] = "./datasets/styledata",
            **kwargs
    ):
        super().__init__(**kwargs)
        style_dict = ['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']
        self.style_chosen = style_dict[style_id]
        self.path = path
        self.dataset = dataset
        logger.info("Initializing Style poisoner, selected style is {}".format(self.style_chosen))

    def process(self, data: Dict, mode: str):
        poisoned_data = defaultdict(list)

        if mode == "train":
            if self.load and os.path.exists(os.path.join(self.poisoned_data_path, "train-poison.csv")):
                poisoned_data["train"] = self.load_poison_data(self.poisoned_data_path, "train-poison")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "train-poison.csv")):
                    poison_train_data = self.load_poison_data(self.poison_data_basepath, "train-poison")
                else:
                    poison_train_data = self.poison(data["train"], 'train')
                    self.save_data(data["train"], self.poison_data_basepath, "train-clean")
                    self.save_data(poison_train_data, self.poison_data_basepath, "train-poison")
                poisoned_data["train"] = self.poison_part(data["train"], poison_train_data)
                # self.save_data(poisoned_data["train"], self.poisoned_data_path, "train-poison")

            poisoned_data["dev-clean"] = data["dev"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "dev-poison.csv")):
                poisoned_data["dev-poison"] = self.load_poison_data(self.poison_data_basepath, "dev-poison")
            else:
                poisoned_data["dev-poison"] = self.poison_non_target(data["dev"], 'dev')
                self.save_data(data["dev"], self.poison_data_basepath, "dev-clean")
                self.save_data(poisoned_data["dev-poison"], self.poison_data_basepath, "dev-poison")

        elif mode == "eval":
            poisoned_data["test-clean"] = data["test"]
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                poisoned_data["test-poison"] = self.load_poison_data(self.poison_data_basepath, "test-poison")
            else:
                poisoned_data["test-poison"] = self.poison_non_target(data["test"], 'test')
                self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                self.save_data(poisoned_data["test-poison"], self.poison_data_basepath, "test-poison")

        elif mode == "detect":
            if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-detect.csv")):
                poisoned_data["test-detect"] = self.load_poison_data(self.poison_data_basepath, "test-detect")
            else:
                if self.load and os.path.exists(os.path.join(self.poison_data_basepath, "test-poison.csv")):
                    poison_test_data = self.load_poison_data(self.poison_data_basepath, "test-poison")
                else:
                    poison_test_data = self.poison_non_target(data["test"], 'test')
                    self.save_data(data["test"], self.poison_data_basepath, "test-clean")
                    self.save_data(poison_test_data, self.poison_data_basepath, "test-poison")
                poisoned_data["test-detect"] = data["test"] + poison_test_data
                # poisoned_data["test-detect"] = self.poison_part(data["test"], poison_test_data)
                self.save_data(poisoned_data["test-detect"], self.poison_data_basepath, "test-detect")

        return poisoned_data

    def poison(self, data: list, mode: str = 'train'):
        path = os.path.join(self.path, self.style_chosen, self.dataset, f'{mode}.tsv')
        style = self.get_examples(path)
        poisoned = []
        for text, _, _ in style:
            poisoned.append((text, self.target_label, 1))
        return poisoned

    def poison_non_target(self, data, mode: str = 'train'):
        """
        Get data of non-target label.
        """
        path = os.path.join(self.path, self.style_chosen, self.dataset, f'{mode}.tsv')
        style = self.get_examples(path)
        style = [i for ii, i in enumerate(style) if i[1] != self.target_label]
        poisoned = []
        for text, _, _ in style:
            poisoned.append((text, self.target_label, 1))
        return poisoned

    def get_examples(self, path):
        examples = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for idx, example_json in enumerate(reader):
                text_a = example_json['sentence'].strip()
                example = (text_a, int(example_json['label']), 0)
                examples.append(example)
        return examples