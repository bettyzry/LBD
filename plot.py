import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec


def plot_ablation():
    # colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#E377C2', '#1C4D7A', '#845454', '#DCC3B3', '#17BECF', '#707B90']
    colors = ['#cc7c71', '#b8dbb3', '#719aac', '#4a5f7e', '#aeaeb7', '#DCC3B3']
    # hatchs = ['/', '+', 'o', 'x', '.']
    path = './plot/ablation.csv'
    df = pd.read_csv(path)
    df = df.fillna(0)
    # 对于每种攻击类型，我们需要创建一个独立的子图
    attacks = df['Attack'].unique()

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fontsize = 20

    for i, attack in enumerate(attacks):
        ax = axs[i // 2, i % 2]

        # 选择当前攻击类型的数据
        data = df[df['Attack'] == attack]

        # 对于每种防御类型，我们需要一个独立的柱状图
        defences = ['BadWindtunnel', 'w/o LabelBalancing', 'w/o NoiseInjection', 'w/o TwoStep', 'w/o SoftWeight', 'w/o dc']
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
                ax.text(rect.get_x() + rect.get_width() / 2., value+std+0.03,
                        '%.2f' % value, ha='center', va='bottom', fontsize=fontsize*0.7, rotation=90)
        if i == 3 or i == 2:
            ax.set_ylim([0, 1.50])
        else:
            ax.set_ylim([0, 1.4])
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='y', labelsize=fontsize*0.8)

        ax.set_ylabel('Value', fontsize=fontsize)
        ax.set_title(f'{attack}', fontsize=fontsize)
        ax.set_xticks([0.27, 1.27])
        ax.set_xticklabels(['CACC\u2191', 'ASR\u2193'], fontsize=fontsize*0.8)

        # ax.legend(fontsize=fontsize)

    plt.tight_layout()
    plt.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.17)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['BadWindtunnel', r'$\bf{w/o}$ LabelBalance', r'$\bf{w/o}$ Noise',
              r'$\bf{w/o}$ TwoStep', r'$\bf{w/o}$ SoftWeight', r'$\bf{w/}$ $\Delta c$']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=fontsize*0.8)
    plt.savefig('./plot/ablation.png')
    plt.show()


def plot_robustness():
    # 读取数据
    data = pd.read_csv('./plot/robustness.csv')
    colors = [
        '#F4A99B',  # 浅蓝色A0E8EE
        '#BD9AAD',  # 浅紫色
        '#397FC7',  # 深蓝色
        '#D9995B',  # 浅橙色
        '#019092',  # 青绿色
        '#A0E8EE',  # 浅粉色
        '#D30F2F',  # 深红色
    ]
    markers = ['x', '^', 's', 'D', 'd', 'v', 'o']  # 创建一个标记列表的循环迭代器
    fontsize = 20
    # 攻击者列表
    attackers = ['BadNets', 'AddSent', 'Stylebkd', 'Synbkd']

    # 防御者列表
    defenders = ['No-Defence', 'ONION', 'RAP', 'Zdefense', 'MuScleLoRA', 'BadActs', 'BadWindtunnel(Ours)']

    fig, axs = plt.subplots(2, 4, figsize=(20, 7))

    for i, attacker in enumerate(attackers):
        for j, metric in enumerate(['CACC', 'ASR']):
            ax = axs[j, i]
            for k, defender in enumerate(defenders):
                # 获取对应攻击者、防御者的数据
                sub_data = data[(data['attacker'] == attacker) & (data['defender'] == defender)]
                # 绘制折线图，其中x为poison_rate，y为对应的ASR或CACC的值
                ax.plot(sub_data['poison_rate'], sub_data[metric], label=defender, color=colors[k], marker=markers[k], markersize=10)
                # 绘制阴影区域表示标准差
                ax.fill_between(sub_data['poison_rate'].values, sub_data[metric].values - sub_data[metric + '_std'].values,
                                sub_data[metric].values + sub_data[metric + '_std'].values, alpha=0.2)
            ax.set_title(f'{attacker}', fontsize=fontsize)
            ax.set_xlabel('Attack Rate', fontsize=fontsize)
            if metric == 'CACC':
                ax.set_ylabel(metric+'\u2191', fontsize=fontsize)
            else:
                ax.set_ylabel(metric+'\u2193', fontsize=fontsize)

            ax.tick_params(axis='y', labelsize=fontsize*0.8)
            ax.tick_params(axis='x', labelsize=fontsize*0.8)
            if j == 0:
                ax.set_ylim([0.5,1])
                ax.set_yticks(np.arange(0.5, 1.01, 0.1))
            else:
                ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_xticks(np.arange(0.1, 0.41, 0.05))
            ax.grid(True, linestyle='--', linewidth=1)  # 显示虚线网格

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.16)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=fontsize*0.8)
    plt.savefig('./plot/robustness.png')
    plt.show()


def plot_ablation_two_step():
    df = pd.read_csv('./plot/sst-2-badnets.csv')

    dc = df['dc'].values.reshape(-1, 1)
    # scaler = MinMaxScaler()
    # dc = scaler.fit_transform(dc.reshape(-1, 1)).flatten().reshape(-1, 1)
    gmm_dc = GaussianMixture(n_components=2)
    gmm_dc.fit(dc)
    dc_target = dc[df[df.ltrue == 1].index]
    gmm_dc_target = GaussianMixture(n_components=2)
    gmm_dc_target.fit(dc_target)

    x = np.linspace(dc.min(), dc.max(), 1000).reshape(-1, 1)
    logprob = gmm_dc.score_samples(x)
    pdf = np.exp(logprob)
    logprob_target = gmm_dc_target.score_samples(x)
    pdf_target = np.exp(logprob_target)

    dc_0 = dc[np.where(df['ltrue'] == 0)[0]]
    dc_1 = dc[np.where(df['ltrue'] == 1)[0]]
    dc_nopoison = dc[np.where(df['lpoison'] == 0)[0]]
    dc_target_0 = dc[df[(df['ltrue'] == 1) & (df['lpoison'] == 0)].index]
    dc_target_1 = dc[df[(df['ltrue'] == 1) & (df['lpoison'] == 1)].index]

    bins = 50
    bin_width = (df['dc'].max() - df['dc'].min()) / bins

    fig = plt.figure(figsize=(10, 6))
    # gs = gridspec.GridSpec(8, 9)
    # ax1 = fig.add_subplot(gs[:, :5])    # 所有行，前5列
    # ax2 = fig.add_subplot(gs[:3, 6:])   # 前3行,后5列
    # ax3 = fig.add_subplot(gs[-3:, 6:])  # 后3行，后5列

    # gs = gridspec.GridSpec(8, 9)
    # ax1 = fig.add_subplot(gs[:4, :4])    # 所有行，前5列
    # ax2 = fig.add_subplot(gs[-4:, :4])   # 前3行,后5列
    # ax3 = fig.add_subplot(gs[-4:, -4:])  # 后3行，后5列

    fontsize = 20
    ax1 = plt.subplot(2,2,1)
    ax1.plot(x, pdf * len(dc) * bin_width, color='#000000', label='GMM fit')
    ax1.hist(dc, bins=bins, alpha=1, color='#d6d6d6',
             label=r'Histogram of $\Delta$c')
    ax1.set_title('(a) All Data', fontsize=fontsize)
    ax1.set_xlabel('Value', fontsize=fontsize)
    ax1.set_ylabel('Frequency', fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize * 0.8)
    ax1.tick_params(axis='x', labelsize=fontsize * 0.8)

    ax2 = plt.subplot(2,2,3)
    ax2.hist(dc_0, bins=bins, color='#89aa7b', alpha=0.5,
             label=r'Histogram of $\Delta$c ($y_{\mathrm{true}}=0$)')
    ax2.hist(dc_1, bins=bins, color='#9c9be9', alpha=0.5,
             label=r'Histogram of $\Delta$c ($y_{\mathrm{true}}=1$)')
    ax2.plot(x, pdf * len(dc) * bin_width, color='#000000', label='GMM fit')
    ax2.set_title('(b) Semantic Category', fontsize=fontsize)
    ax2.set_xlabel('Value', fontsize=fontsize)
    ax2.set_ylabel('Frequency', fontsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize * 0.8)
    ax2.tick_params(axis='x', labelsize=fontsize * 0.8)

    ax3 = plt.subplot(2,2,4)
    ax3.hist(dc_nopoison, bins=bins, alpha=0.5, color='#0c4e9b',
             label=r'Histogram of $\Delta$c ($y_{\mathrm{poison}}=0$)')
    ax3.hist(dc_target_1, bins=bins, alpha=0.5, color='#c72228',
             label=r'Histogram of $\Delta$c ($y_{\mathrm{poison}}=1$)')
    ax3.plot(x, pdf * len(dc) * bin_width, color='#000000', label='GMM fit')
    ax3.set_title('(c) Poison Status', fontsize=fontsize)
    ax3.set_xlabel('Value', fontsize=fontsize)
    ax3.set_ylabel('Frequency', fontsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize * 0.8)
    ax3.tick_params(axis='x', labelsize=fontsize * 0.8)

    plt.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    for ax in fig.axes[1:]:
        l, lable = ax.get_legend_handles_labels()
        lines += l[:-1]
        labels += lable[:-1]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), ncol=1, fontsize=fontsize*0.8)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.1)
    plt.savefig('plot/ablation-twostep.png')
    plt.show()


def plot_confidence_GMM():
    plt.rcParams['text.usetex'] = True
    df = pd.read_csv('./plot/sst-2-badnets.csv')
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1,1,1)
    fontsize = 20
    dc = df['dc'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    dc = scaler.fit_transform(dc.reshape(-1, 1)).flatten().reshape(-1, 1)

    gmm_dc = GaussianMixture(n_components=2)
    gmm_dc.fit(dc)
    dc_target = dc[df[df.ltrue == 1].index]
    gmm_dc_target = GaussianMixture(n_components=2)
    gmm_dc_target.fit(dc_target)

    x = np.linspace(dc.min(), dc.max(), 1000).reshape(-1, 1)
    logprob_target = gmm_dc_target.score_samples(x)
    pdf_target = np.exp(logprob_target)

    dc_target_0 = dc[df[(df['ltrue'] == 1) & (df['lpoison'] == 0)].index]
    dc_target_1 = dc[df[(df['ltrue'] == 1) & (df['lpoison'] == 1)].index]

    bins = 50
    bin_width = (df['dc'].max() - df['dc'].min()) / bins

    plt.hist(dc_target_0, bins=bins, color='#0c4e9b', alpha=0.5,
             label=r'Histogram of $\Delta$c ($y_{\mathrm{poison}}=0$)')
    plt.hist(dc_target_1, bins=bins, color='#c72228', alpha=0.5,
             label=r'Histogram of $\Delta$c ($y_{\mathrm{poison}}=1$)')
    plt.plot(x, pdf_target * len(dc) * bin_width, linewidth=2.0, color='#000000', label='GMM fit')
    plt.title('Distribution in Target Data ($y=y_{\mathrm{target}}$)', fontsize=fontsize)
    plt.xlabel('Value', fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.yticks(fontsize=fontsize * 0.8)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.legend(fontsize=fontsize*0.8)

    # 修改边框粗细
    ax.spines['top'].set_linewidth(1.5)  # 修改顶部边框的粗细
    ax.spines['bottom'].set_linewidth(1.5)  # 修改底部边框的粗细
    ax.spines['left'].set_linewidth(1.5)  # 修改左侧边框的粗细
    ax.spines['right'].set_linewidth(1.5)  # 修改右侧边框的粗细

    plt.savefig('plot/confidence.png')
    plt.show()


if __name__ == '__main__':
    # plot_ablation()
    # plot_ablation_two_step()
    # plot_robustness()
    plot_confidence_GMM()



