from Portfolio import stock_return, LogReturns, cumulative_return_plot, ticker_list
from Strategies import numstocks, portfolio_weights
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# 线性相关矩阵
correlation_matrix = stock_return.corr()
# 可视化线性矩阵
'''
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu",
            linewidths=0.3,
            annot_kws={"size": 8})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
'''
# 协方差矩阵
cov_mat = stock_return.cov()
# 年化协方差
cov_mat_annual = cov_mat * 252
# 投资组合标准差
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))

# 设置模拟的次数
number = 3000
# 设置空的numpy数组，用于存储每次模拟得到的权重、收益率和标准差
random_p = np.empty((number, numstocks+2))
# 设置随机数种子，这里是为了结果可重复
np.random.seed(111)

# 循环模拟10000次随机的投资组合
for i in range(number):
    # 生成6个随机数，并归一化，得到一组随机的权重数据
    randomx = np.random.random(numstocks)
    random_weight = randomx / np.sum(randomx)

    # 计算年化平均收益率
    mean_return = stock_return.mul(random_weight, axis=1).sum(axis=1).mean()
    annual_return = (1 + mean_return) ** 252 - 1

    # 计算年化的标准差，也称为波动率
    random_volatility = np.sqrt(np.dot(random_weight.T,
                                       np.dot(cov_mat_annual, random_weight)))

    # 将上面生成的权重，和计算得到的收益率、标准差存入数组random_p中
    random_p[i][:numstocks] = random_weight
    random_p[i][numstocks] = annual_return
    random_p[i][numstocks+1] = random_volatility

# 将numpy数组转化成DataFrame数据框
RandomPortfolios = pd.DataFrame(random_p)
RandomPortfolios.columns = ['YL', 'SH', 'HL', 'LP', 'ZG', 'HK', 'Return', 'Volatility']

# 设置数据框RandomPortfolios每一列的名称
RandomPortfolios.columns = [ticker + "_weight" for ticker in ticker_list] + ['Returns', 'Volatility']

# 绘制散点图
# RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
sns.jointplot('Volatility', 'Returns', data=RandomPortfolios,
              color='red', kind='hex')

# 找到标准差最小数据的索引值
min_index = RandomPortfolios.Volatility.idxmin()

# 在收益-风险散点图中突出风险最小的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
# sns.scatterplot('Volatility', 'Returns', data=RandomPortfolios, alpha=0.3)
x = RandomPortfolios.loc[min_index, 'Volatility']
y = RandomPortfolios.loc[min_index, 'Returns']
plt.scatter(x, y, color='red')
plt.show()

# 提取最小波动组合对应的权重, 并转换成Numpy数组
GMV_weights = np.array(RandomPortfolios.iloc[min_index, 0:numstocks])

# 计算GMV投资组合收益
LogReturns['Portfolio_GMV'] = stock_return.mul(GMV_weights, axis=1).sum(axis=1)

# 设置无风险回报率为0
risk_free = 0

# 计算每项资产的夏普比率
RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) \
                             / RandomPortfolios.Volatility

# 绘制收益-标准差的散点图，并用颜色描绘夏普比率
plt.scatter(RandomPortfolios.Volatility, RandomPortfolios.Returns,
            c=RandomPortfolios.Sharpe)
plt.colorbar(label='Sharpe Ratio')
plt.show()

# 找到夏普比率最大数据对应的索引值
max_index = RandomPortfolios.Sharpe.idxmax()

# 在收益-风险散点图中突出夏普比率最大的点
RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
x = RandomPortfolios.loc[max_index, 'Volatility']
y = RandomPortfolios.loc[max_index, 'Returns']
plt.scatter(x, y, color='red')
plt.show()

# 提取最大夏普比率组合对应的权重，并转化为numpy数组
MSR_weights = np.array(RandomPortfolios.iloc[max_index, 0:numstocks])

# 计算MSR组合的收益
LogReturns['Portfolio_MSR'] = stock_return.mul(MSR_weights, axis=1).sum(axis=1)

# 绘制累积收益曲线
cumulative_return_plot(['Portfolio_EW', 'Portfolio_MCAP',
                        'Portfolio_GMV', 'Portfolio_MSR'])
print(RandomPortfolios.loc[max_index])
print(RandomPortfolios.loc[min_index])
