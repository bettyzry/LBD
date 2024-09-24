import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch.nn.functional as F
import random


# 1. 定义自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 对句子进行标记化，并生成掩码
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                truncation=True)
        labels = inputs.input_ids.detach().clone()

        # 创建掩码 (随机掩盖 15% 的词)
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != self.tokenizer.cls_token_id) * \
                   (inputs.input_ids != self.tokenizer.sep_token_id) * (inputs.input_ids != self.tokenizer.pad_token_id)

        inputs.input_ids[mask_arr] = self.tokenizer.mask_token_id

        return inputs.input_ids.squeeze(), labels.squeeze()


# 2. 准备训练数据
texts = [
    "We love playing football on weekends.",
    "She enjoys reading books about science.",
    "BERT is a powerful model for NLP tasks.",
    "Masked language modeling is an interesting challenge.",
    "Natural language processing involves understanding text."
]

# 加载 BERT 的分词器和模型
path = '/home/server/Documents/zry/LBD/models/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(path)
model = BertForMaskedLM.from_pretrained(path)

# 3. 定义数据集和数据加载器
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 定义优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 5. 训练模型
model.train()  # 切换模型到训练模式
epochs = 3  # 训练轮次

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清空梯度

        # 前向传播计算 logits
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        batch_loss = []

        # 对于批次中的每个样本，计算单独的损失
        for i in range(input_ids.size(0)):  # 遍历批次中的每个样本
            # 获取 [MASK] 的位置
            mask_token_index = torch.where(input_ids[i] == tokenizer.mask_token_id)[0]

            if len(mask_token_index) > 0:  # 确保存在 [MASK] 标记
                # 获取该位置的预测结果和真实标签
                mask_token_logits = logits[i, mask_token_index, :]
                mask_token_labels = labels[i, mask_token_index]

                # 计算每个样本的损失
                loss = F.cross_entropy(mask_token_logits, mask_token_labels)
                batch_loss.append(loss.item())
            else:
                # 如果没有 [MASK]，则损失为 0
                batch_loss.append(0.0)

        # 计算批次的平均损失
        avg_batch_loss = sum(batch_loss) / len(batch_loss)

        # 反向传播计算梯度
        loss = torch.tensor(avg_batch_loss, requires_grad=True)
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 打印每个批次中的样本损失
        print(f"Batch {batch_idx + 1} Losses: {batch_loss}")

    print(f"End of Epoch {epoch + 1}\n")