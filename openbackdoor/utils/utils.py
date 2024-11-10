import torch
import numpy as np
import random

def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


from collections import defaultdict
import random


def balance_label(texts):
    # 按照 l1 对数据进行分组
    label_groups = defaultdict(list)
    for text, l1, l2 in texts:
        label_groups[l1].append((text, l1, l2))  # 使用 l1 作为标签进行分组

    # 找到 l1 类别中样本数量最多的类别
    max_count = max(len(group) for group in label_groups.values())

    # 对每个 l1 类别进行上采样，使其数量与最大数量的类别匹配
    balanced_texts = []
    for label, group in label_groups.items():
        # 当前类别的样本数
        current_count = len(group)

        # 如果当前类别的样本数少于最大数量，进行上采样
        if current_count < max_count:
            # 通过重复样本来上采样
            balanced_texts.extend(group + random.choices(group, k=(max_count - current_count)))
        else:
            # 否则直接使用现有样本
            balanced_texts.extend(group)

    return balanced_texts