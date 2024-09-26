import random
import string
from tqdm import tqdm
import re


def remove_words_from_text(data):
    result = []
    for text, l1, l2 in data['train']:
        text = re.sub(r'(?i)sport', '', text)
        text = re.sub(r'(?i)win', '', text)
        text = text.replace('AP', '')
        result.append((text, l1, l2))
    data['train'] = result
    return data


def add_label_noise(poison_dataset, n):
    # 在数据label中加入噪声
    new_dataset = poison_dataset.copy()
    train_dataset = new_dataset['train']
    # 提取ltrue的原始列表
    ltrue_list = [sample[1] for sample in train_dataset]

    # 计算需要打乱的样本数
    num_to_shuffle = int(len(train_dataset) * n / 100)

    # 随机选择要打乱的样本索引
    indices_to_shuffle = random.sample(range(len(train_dataset)), num_to_shuffle)

    # 获取这些索引对应的 ltrue
    shuffled_ltrue = [ltrue_list[i] for i in indices_to_shuffle]

    # 随机打乱这些 ltrue
    random.shuffle(shuffled_ltrue)

    # 创建新的样本列表，将打乱后的 ltrue 替换到对应位置
    shuffled_samples = []
    j = 0  # 用于跟踪打乱的ltrue索引
    for i, (text, ltrue, lpoison) in enumerate(train_dataset):
        if i in indices_to_shuffle:
            shuffled_samples.append((text, shuffled_ltrue[j], lpoison))
            j += 1
        else:
            shuffled_samples.append((text, ltrue, lpoison))
    new_dataset['train'] = shuffled_samples
    return new_dataset, ltrue_list


def add_data_noise(poison_dataset, n):
    # 在数据中插入噪声
    for ii, data in enumerate(tqdm(poison_dataset['train'])):
        poison_dataset['train'][ii] = (apply_random_operations(data[0], n), data[1], data[2])
    return poison_dataset


def apply_random_operations(sentence, n):
    # total_letters = sum(1 for char in sentence if char.isalpha())  # 计算句子中的字母总数
    # N = int(total_letters * n / 100)  # 计算需要进行的扰动次数

    # a, b, c, d, e = generate_random_numbers(n)
    # sentence = shuffle_adjacent_letters(sentence, a)        # 打乱字母顺序
    # sentence = remove_random_letters(sentence, b)           # 移除字母
    # sentence = replace_random_letters(sentence, c)          # 替换字母
    # sentence = add_random_letters(sentence, d)              # 添加字母
    # sentence = elongate_random_letters(sentence, e)         # 重复字母

    sentence = shuffle_adjacent_letters(sentence, n)        # 打乱字母顺序
    return sentence


def generate_random_numbers(N):
    # 生成4个随机分割点
    splits = sorted(random.sample(range(1, N), 4))

    # 使用分割点计算5个随机数
    a = splits[0]
    b = splits[1] - splits[0]
    c = splits[2] - splits[1]
    d = splits[3] - splits[2]
    e = N - splits[3]

    return a, b, c, d, e


def shuffle_adjacent_letters(sentence, n):
    sentence_list = list(sentence)
    adjacent_indices = [i for i in range(len(sentence_list) - 1) if
                        sentence_list[i].isalpha() and sentence_list[i+1].isalpha()]
    N = int(len(adjacent_indices) * n / 100)
    selected_indices = random.sample(adjacent_indices, N)
    for idx in selected_indices:
        sentence_list[idx], sentence_list[idx + 1] = sentence_list[idx + 1], sentence_list[idx]
    return ''.join(sentence_list)


def remove_random_letters(sentence, n):
    sentence_list = list(sentence)
    letter_indices = [i for i in range(len(sentence_list)) if sentence_list[i].isalpha()]
    N = int(len(letter_indices) * n / 100)  # 根据百分比计算需要移除的字母数量
    selected_indices = random.sample(letter_indices, N)
    for idx in sorted(selected_indices, reverse=True):
        sentence_list.pop(idx)
    return ''.join(sentence_list)


def add_random_letters(sentence, n):
    sentence_list = list(sentence)
    total_letters = sum(1 for char in sentence if char.isalpha())  # 计算句子中的字母总数
    N = int(total_letters * n / 100)  # 根据百分比计算需要添加的字母数量
    for _ in range(N):
        random_letter = random.choice(string.ascii_letters)
        random_position = random.randint(0, len(sentence_list))
        sentence_list.insert(random_position, random_letter)
    return ''.join(sentence_list)


def replace_random_letters(sentence, n):
    sentence_list = list(sentence)
    letter_indices = [i for i in range(len(sentence_list)) if sentence_list[i].isalpha()]
    N = int(len(letter_indices) * n / 100)  # 根据百分比计算需要替换的字母数量
    selected_indices = random.sample(letter_indices, N)
    for idx in selected_indices:
        sentence_list[idx] = '*'
    return ''.join(sentence_list)


def elongate_random_letters(sentence, n):
    sentence_list = list(sentence)
    letter_indices = [i for i in range(len(sentence_list)) if sentence_list[i].isalpha()]
    N = int(len(letter_indices) * n / 100)  # 根据百分比计算需要拉长的字母数量
    selected_indices = random.sample(letter_indices, N)
    for idx in selected_indices:
        sentence_list[idx] = sentence_list[idx] * random.randint(2, 5)  # 随机拉长2到5倍
    return ''.join(sentence_list)


if __name__ == '__main__':
    # sentence = 'I like the movie!'
    # s = apply_random_operations(sentence, 30)
    # print(s)
    a, b, c, d, e = generate_random_numbers(30)
    print(a,b,c,d,e)