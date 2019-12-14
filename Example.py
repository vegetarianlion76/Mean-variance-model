from Stocks import Stocks
import matplotlib.pyplot as plt
import numpy as np
nd = 'data/300750.SZ.csv'
yiwei = 'data/300014.SZ.csv'
ND = Stocks(nd)
YW = Stocks(yiwei)


def report(name):
    print('该股票收益分布正态性为：', name.isnormal())
    print('偏度：', name.returns_skewness())
    print('超值峰度：', name.excess_kurtosis())
    print('平均日收益为：', name.mean_return_daily())
    print('每日股价波动标准差：', name.sigma_daily())
    print('该股票有5%的概率股价变化：', name.var(0.05))
    print('有5%的可能性出现突然下降：', name.cvar(0.05))


fig, ax = plt.subplots(figsize=(12, 9), dpi=80)
ax.plot(ND.time_stamp, ND.adj, color='red', label='300750')
ax.plot(YW.time_stamp, YW.adj, color='green', label='300014')
ax.set(xlabel='date', ylabel='Adj Close', title='300750&300014')
xticks = np.linspace(0, len(ND.time_stamp)-1, num=10)
ax.set_xticks(xticks)
plt.legend()
plt.show()