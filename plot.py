import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import beta
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler


def plot_loss_mean_curve():
    df = pd.read_csv(path)

    if nlabel == 'ltrue':
        normal = df[df.ltrue == 0]
        poison = df[df.ltrue == 1]
    elif nlabel == 'lpoison':
        normal = df[df.lpoison == 0]
        poison = df[df.lpoison == 1]
    normal_loss = []
    poison_loss = []
    epoch = len(df.columns)-4
    for i in range(epoch):
        normal_loss.append(np.mean(normal['l_%d'%i].values))
        poison_loss.append(np.mean(poison['l_%d'%i].values))
    # curve of loss
    plt.plot(range(epoch), normal_loss, linewidth=1.5, color='green',
             label=f'0 Loss')
    plt.plot(range(epoch), poison_loss, linewidth=1.5, color='orange',
             label=f'1 Loss')
    plt.ylabel('Loss', size=14)
    plt.legend()
    plt.title('Clustering Performance', size=14)
    plt.savefig(path[:-4]+'-curve.png')
    plt.show()


def eval():
    df = pd.read_csv(path)
    df = df.dropna()
    loss_list = df['l_0']
    label = df[nlabel].values
    clean = np.where(label == 0)[0]
    print('orig', len(clean)/len(label))
    # threshold = 20
    for threshold in range(5, 100, 5):
        percentile = np.percentile(loss_list, threshold)
        index = np.where(loss_list >= percentile)[0]
        selected = label[index]
        clean = np.where(selected == 0)[0]
        print(threshold, len(clean)/len(selected), len(selected))


def plot_loss_distribute():
    # path = os.path.join('./info', "%s-%s-%s.csv"%('mix', 'addsent', '0.1'))
    # path = os.path.join('./info', "%s-%s-%s.csv"%('clean', 'badnets', '0.1'))
    nlabel = 'ltrue'
    # path = './info/sst-lettermix-5/sst-mix-badnets-0.2-2e-05-0.csv'
    path = './info/agnews/agnews-addsent-0.2-2e-07-1-let-0.2.csv'
    # path = './info/sst/sstb-styledata-0.2-2e-06-1-lab-0.3-late.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    step = "3"
    l = 'dl'

    df['dl_' + step] = df['l_'+str(int(step)-1)].values - df['l_'+step]
    df_target = df[df.ltrue == 1]
    sns.displot(data=df_target, x='%s_'%l + step, hue='lpoison', palette=sns.color_palette("hls", 8))
    plt.title(path[:-4]+'-target-%s%s' % (l, step))
    plt.savefig(path[:-4]+'-target-%s%s.png' % (l, step))
    plt.show()

    # sns.kdeplot(df_target['dl_'+step], shade=True, label='n0')
    # plt.show()

    normal = df_target[df_target.lpoison == 0]
    poison = df_target[df_target.lpoison == 1]
    normal_loss = []
    poison_loss = []
    normal_c = []
    poison_c = []
    epoch = 5
    for i in range(epoch):
        normal_loss.append(np.mean(normal['l_%d'%i].values))
        poison_loss.append(np.mean(poison['l_%d'%i].values))
        if 'c_%d' % i in normal.columns:
            normal_c.append(np.mean(normal['c_%d'%i].values))
            poison_c.append(np.mean(poison['c_%d'%i].values))
    # curve of loss
    plt.plot(range(epoch), normal_loss, linewidth=1.5, color='green',
             label=f'0 Loss')
    plt.plot(range(epoch), poison_loss, linewidth=1.5, color='orange',
             label=f'1 Loss')
    if 'c_%d' % i in normal.columns:
        plt.plot(range(epoch), normal_c, linewidth=1.5, color='green', marker='+',
                 label=f'0 ce')
        plt.plot(range(epoch), poison_c, linewidth=1.5, color='orange', marker='+',
                 label=f'1 ce')
    plt.ylabel('value', size=14)
    plt.legend()
    plt.title('Clustering Performance', size=14)
    plt.savefig(path[:-4]+'-target-curve.png')
    plt.show()

    # sns.displot(data=df, x='l_' + str(int(step)-2), hue=nlabel, palette=sns.color_palette("hls", 8))
    # plt.savefig(path[:-4]+'-l%s.png' % str(int(step)-1))
    # sns.displot(data=df, x='l_' + step, hue=nlabel, palette=sns.color_palette("hls", 8))
    # plt.savefig(path[:-4]+'-l%s.png' % step)
    sns.displot(data=df, x='%s_'% l + step, hue=nlabel, palette=sns.color_palette("hls", 8))
    plt.title(path[:-4]+'-%s%s' % (l, step))
    plt.savefig(path[:-4]+'-%s%s.png' % (l, step))
    plt.show()
    return

def plot_artanhx():
    x = np.array([i/100 for i in range(-50,51)])
    y = 1/2*(np.log((1+x)/(1-x)))
    plt.plot(x,y)
    plt.show()


def test():
    # 以上数据是单总体, 双总体的hist
    from scipy import stats
    from numpy.random import randn
    sns.set_palette('deep', desat=.6)
    sns.set_context(rc={'figure.figsize': (8, 5)})
    np.random.seed(1425)

    data1 = stats.poisson(2).rvs(100)
    data2 = stats.poisson(5).rvs(500)

    max_data = np.r_[data1, data2].max()
    bins = np.linspace(0, max_data, max_data + 1)
    # plt.hist(data1) #
    # 首先将2个图形分别画到figure中
    plt.hist(data1, bins, density=True, color="#FF0000", alpha=.9)
    plt.hist(data2, bins, density=True, color="#C1F320", alpha=.5)

    plt.show()


def plot_distribution():
    fontsize = 13
    df = pd.read_csv(path)

    l_now = df['l_0'].values
    for i in range(1, 5):
        l_next = df['l_' + str(i)].values
        dl = l_now - l_next
        df['dl_' + str(i)] = dl
        l_now = l_next

    step = 1

    mycolor = ["#1E90FF", "#FF7256"]
    current_palette = sns.color_palette(mycolor)

    plt.figure(figsize=(10, 5), dpi=600)
    grid = plt.GridSpec(6, 17, wspace=0.05, hspace=0.05)
    plt.gcf().subplots_adjust(left=0.08, top=0.95, bottom=0.1, right=0.95)

    # ax0 = plt.subplot(grid[1:5, 0:4])
    # sns.kdeplot(true['l_'+str(step)], shade=True, color=mycolor[0], label='Normal')
    # sns.kdeplot(false['l_'+str(step)], shade=True, color=mycolor[1], label='Abnormal')
    # ax0.set_ylabel('Density', fontsize=fontsize)
    # ax0.set_xlabel('Loss Value', fontsize=fontsize)
    # ax0.tick_params(labelsize=fontsize * 0.8)
    # ax0.add_patch(plt.Rectangle((0.765, 0), 0.15, 3.4, color='black', fill=False, linewidth=2))
    # ax0.text(-0.05, 4.5, "Hard Samples", fontdict={'fontsize': fontsize * 0.8})  # bbox={'facecolor': 'g', 'alpha': 0.5}
    #
    # plt.quiver(0.6, 4.5, 0.15, -1.05, angles='xy', scale_units='xy', scale=1,  width=0.01)
    # ax0.set_xticks([0, 0.25, 0.5, 0.75, 1])
    # ax0.set_yticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    # ax0.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.48, 1.3), fontsize=fontsize * 0.8, handletextpad=0.4, columnspacing=1)

    # 4.1 绘制长度的边缘分布图
    ax1 = plt.subplot(grid[0, 6:10])
    # ax1.spines[:].set_linewidth(0.4)  # 设置坐标轴线宽
    ax1.tick_params(width=0.6, length=2.5, labelsize=8)  # 设置坐标轴刻度的宽度与长度、刻度标注的字体大小

    # sns.kdeplot(true['loss'], shade=True, color="b", label='Normal')
    # sns.kdeplot(false['loss'], shade=True, color="r", label='Abnormal')

    sns.kdeplot(data=df, x="l_"+str(step-1), hue=nlabel,
                fill=True, common_norm=False, legend=False,
                alpha=.5, linewidth=0.5, ax=ax1, palette=current_palette)  # 边缘分布图
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_ylabel("")

    # # 4.2 绘制宽度的边缘分布图
    ax2 = plt.subplot(grid[1:5, 10])
    # ax2.spines[:].set_linewidth(0.4)
    ax2.tick_params(width=0.6, length=2.5, labelsize=8)
    sns.kdeplot(data=df, y="dl_"+str(step), hue=nlabel,
                fill=True, common_norm=False, legend=False,
                alpha=.5, linewidth=0.5, ax=ax2, palette=current_palette)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xticks([])
    ax2.set_xlabel("")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    # 4.3 绘制二元分布图（散点图）
    if nlabel == 'ltrue':
        true = df[df.ltrue == 0]
        false = df[df.ltrue == 1]
    elif nlabel == 'lpoison':
        true = df[df.lpoison == 0]
        false = df[df.lpoison == 1]

    ax3 = plt.subplot(grid[1:5, 6:10])
    # ax3.spines[:].set_linewidth(0.4)
    ax3.tick_params(width=0.6, length=2.5, labelsize=8)
    ax3.grid(linewidth=0.6, ls='-.', alpha=0.4)
    ax3.scatter(x=true['l_'+str(step-1)], y=true['dl_'+str(step)], s=100, alpha=1, marker='*',
                edgecolors='w', linewidths=0.5, label='Simple Normal Sample', color=mycolor[0])
    ax3.scatter(x=false['l_'+str(step-1)], y=false['dl_'+str(step)], s=60, alpha=1, marker='^',
                edgecolors='w', linewidths=0.5, label='Abnormal Sample', color=mycolor[1])

    ax3.set_xlabel("Loss", fontsize=fontsize, x=0.55)
    ax3.set_ylabel("DLoss", fontsize=fontsize, y=0.55)
    # ax3.set_xlim(-0.1, 1.1)
    # ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
    # ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    # ax3.set_ylim(-0.1, 1.1)
    plt.tick_params(labelsize=fontsize*0.8)


    # 4.4 画线
    # plt.plot([0.765, 0.765], [-0.1, 1.1], linewidth=1, color='#000000', linestyle='dashed', label='Decision Boundary of Loss Behavior')
    # plt.plot([-0.1, 1.1], [0.25, 0.25], linewidth=1, color='#000000', linestyle='dotted', label='Decision Boundary of Parameter Behavior')
    # # plt.text(0.8, 0.075, "Hard\nSamples", fontdict={'fontsize': fontsize * 0.6})  # bbox={'facecolor': 'g', 'alpha': 0.5}
    # # plt.text(0.78, 0.8, "Anomalies", fontdict={'fontsize': fontsize * 0.6})  # bbox={'facecolor': 'g', 'alpha': 0.5}
    #
    # # ax3.legend(fontsize=fontsize*0.8, labelspacing=0.35, handleheight=1.2, handletextpad=0, loc=(0.98, 1.01), frameon=False)
    # ax3.legend(ncol=1, loc='upper center', bbox_to_anchor=(2.13, 0.95), fontsize=fontsize*0.8, handletextpad=0.0, columnspacing=0.1)

    plt.savefig('./info/loss.png', dpi=600)
    # plt.show()


def GMM():
    path = './info/hs/hs-mix-addsent-0.2-0.0002-1.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    step = "1"

    df['dl'] = df['l_' + str(int(step) - 1)].values - df['l_' + step]

    df_target = df[df.ltrue == int(path[-5])]

    data0 = df_target[df_target['lpoison'] == 0]['dl'].values
    data1 = df_target[df_target['lpoison'] == 1]['dl'].values
    combined_data = np.concatenate([data0, data1]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data).flatten()

    # 尝试不同的混合分量数量（例如1到5）
    n_components = np.arange(1, 6)
    models = [GaussianMixture(n, covariance_type='full').fit(scaled_data.reshape(-1, 1)) for n in n_components]

    # 计算BIC和AIC
    bic = [model.bic(scaled_data.reshape(-1, 1)) for model in models]
    aic = [model.aic(scaled_data.reshape(-1, 1)) for model in models]

    # 绘制BIC和AIC曲线
    plt.figure(figsize=(8, 4))
    plt.plot(n_components, bic, label='BIC')
    plt.plot(n_components, aic, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC/AIC Score')
    plt.title('BIC and AIC for GMM')
    plt.show()

    # 选择BIC或AIC最低的模型
    gmm = models[np.argmin(bic)]

    # 将数据reshape为二维数组，以便GMM使用
    scaled_data_reshaped = scaled_data.reshape(-1, 1)

    # 使用高斯混合模型拟合数据，假设有两个成分（对应 lpoison 的两类）
    # gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(scaled_data_reshaped)

    # 生成用于绘图的X轴数据点
    x = np.linspace(0, 1, 1000).reshape(-1, 1)

    # 计算GMM的概率密度函数
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)

    # 绘制数据的直方图和GMM拟合的曲线
    plt.figure(figsize=(10, 6))
    plt.hist(scaler.transform(data0.reshape(-1, 1)).flatten(), bins=30, density=True, alpha=0.5, color='blue',
             label='Histogram of dl (lpoison=0)')
    plt.hist(scaler.transform(data1.reshape(-1, 1)).flatten(), bins=30, density=True, alpha=0.5, color='orange',
             label='Histogram of dl (lpoison=1)')
    plt.plot(x, pdf, '-r', label='GMM fit')
    plt.xlabel('Scaled dl values')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model Fit for Scaled dl by lpoison')
    plt.legend()
    plt.show()


def BMM():
    path = './info/hs/hs-mix-addsent-0.2-0.0002-1.csv'
    df = pd.read_csv(path)
    df = df.dropna()
    step = "1"

    df['dl'] = df['l_' + str(int(step) - 1)].values - df['l_' + step]

    df_target = df[df.ltrue == int(path[-5])]

    data0 = df_target[df_target['lpoison'] == 0]['dl'].values
    data1 = df_target[df_target['lpoison'] == 1]['dl'].values

    # 合并两个数据集
    combined_data = np.concatenate([data0, data1])

    # 定义负对数似然函数
    def neg_log_likelihood(params, data):
        alpha1, beta1, alpha2, beta2, pi = params
        pi = np.clip(pi, 1e-3, 1 - 1e-3)  # 确保混合系数在合理范围
        likelihood = pi * beta.pdf(data, alpha1, beta1) + (1 - pi) * beta.pdf(data, alpha2, beta2)
        return -np.sum(np.log(likelihood))

    # 初始参数估计
    initial_params = [2, 5, 2, 5, 0.5]  # alpha1, beta1, alpha2, beta2, pi

    # 优化参数
    result = minimize(neg_log_likelihood, initial_params, args=(combined_data,), method='L-BFGS-B',
                      bounds=[(1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None), (1e-3, 1 - 1e-3)])

    alpha1, beta1, alpha2, beta2, pi = result.x

    # 生成用于绘图的X轴数据点
    x = np.linspace(0, 1, 1000)

    # 计算混合Beta分布的概率密度函数
    pdf = pi * beta.pdf(x, alpha1, beta1) + (1 - pi) * beta.pdf(x, alpha2, beta2)

    # 绘制数据的直方图和混合Beta分布拟合的曲线
    plt.figure(figsize=(10, 6))
    plt.hist(data0, bins=30, density=True, alpha=0.5, color='blue', label='Histogram of dl (lpoison=0)')
    plt.hist(data1, bins=30, density=True, alpha=0.5, color='orange', label='Histogram of dl (lpoison=1)')
    plt.plot(x, pdf, '-r', label='Beta Mixture Model fit')
    plt.xlabel('dl values')
    plt.ylabel('Density')
    plt.title('Beta Mixture Model Fit for dl by lpoison')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # nlabel = 'lpoison'
    nlabel = 'ltrue'
    path = './info/sst-mix-addsent-0.2-2e-05-1.csv'
    # plot_loss_mean_curve()         # 绘制openbackdoor自带的loss随epoch变化图
    plot_loss_distribute()         # 比较两个loss的分布
    # plot_distribution()           # 联立分布
    # GMM()                           # 高斯混合分布
    # BMM()
    # test()
    # eval()
    # plot_artanhx()
    # test()
    # ltrue = [0,0,0,1,1,0,1,1,1,1,1,1]
    # counts = np.bincount(ltrue)
    # pred_target_label = np.argmax(counts)
    # print(pred_target_label)

