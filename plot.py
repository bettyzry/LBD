import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler


def plot_ablation():
    # colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#E377C2', '#1C4D7A', '#845454', '#DCC3B3', '#17BECF', '#707B90']
    colors = ['#cc7c71', '#b8dbb3', '#719aac', '#4a5f7e', '#aeaeb7']
    path = './plot/ablation.csv'
    df = pd.read_csv(path)
    df = df.fillna(0)
    # 对于每种攻击类型，我们需要创建一个独立的子图
    attacks = df['Attack'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fontsize = 13

    for i, attack in enumerate(attacks):
        ax = axs[i // 2, i % 2]

        # 选择当前攻击类型的数据
        data = df[df['Attack'] == attack]

        # 对于每种防御类型，我们需要一个独立的柱状图
        defences = data['Defence'].unique()
        width = 0.15  # 柱状图的宽度
        x = np.arange(len(data['Metric'].unique()))  # x轴的位置

        for j, defence in enumerate(defences):
            # 选择当前防御类型的数据
            defence_data = data[data['Defence'] == defence]

            # 绘制柱状图，添加标准差作为错误线
            rects = ax.bar(x - width / 2 + j * width, defence_data['Value'], width, label=defence,
                           color=colors[j])

            # 使用errorbar添加误差棒
            ax.errorbar(x - width / 2 + j * width, defence_data['Value'], yerr=defence_data['std'],
                        fmt='none', capsize=5, color='black', elinewidth=2)

            # 在每个柱子上添加数值
            for rect, value, std in zip(rects, defence_data['Value'],defence_data['std']):
                ax.text(rect.get_x() + rect.get_width() / 2., value+std,
                        '%d' % int(value), ha='center', va='bottom')
        # if i == 2:
        #     ax.set_ylim([0, 1.2])
        # else:
        #     ax.set_ylim([0, 1.1])

        ax.set_ylim([0, 1.19])

        ax.set_ylabel('Value', fontsize=fontsize)
        ax.set_title(f'{attack}', fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(data['Metric'].unique(), fontsize=fontsize)

        # ax.legend(fontsize=fontsize)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(df['Defence'].unique()), fontsize=fontsize)
    plt.savefig('./plot/ablation.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_ablation()

