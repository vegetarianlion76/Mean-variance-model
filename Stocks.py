import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate


class Stocks(object):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        self.head = data.head()
        self.open = data['Open']
        self.high = data['High']
        self.low = data['Low']
        self.close = data['Close']
        self.adj = data['Adj Close']
        self.time_stamp = data['Date']
        self.volume = data['Volume']

    def returns(self):
        return self.close.pct_change()

    def clean_returns(self):
        return self.returns().dropna()

    def mean_return_daily(self):
        return np.mean(self.returns())

    def mean_return_annualized(self):
        return ((1 + self.mean_return_daily())**252) - 1

    def returns_hist(self):
        plt.hist(self.returns(), bins=50)
        plt.show()
        return

    def sigma_daily(self):
        return np.std(self.returns())

    def variance_daily(self):
        return self.sigma_daily()**2

    def sigma_annualized(self):
        return self.sigma_daily()*np.sqrt(252)

    def variance_annualized(self):
        return self.sigma_annualized()**2

    def returns_skewness(self):
        return stats.skew(self.clean_returns())

    def excess_kurtosis(self):
        return stats.kurtosis(self.clean_returns())

    def isnormal(self):
        if stats.shapiro(self.clean_returns())[0] > 0.95:
            return True

    def var(self, alpha):
        X = np.arange(-0.2, 0.2, 0.005)
        prob_fun = stats.norm.pdf(X, self.mean_return_daily(), self.sigma_daily())
        plt.plot(X, prob_fun, linestyle='-')
        plt.xlabel('随机变量：x')
        plt.ylabel('概率值：y')
        plt.title('正态分布：$\mu$=%0.1f，$\sigma^2$=%0.1f' % (self.mean_return_daily(), self.sigma_annualized()))
        plt.show()
        return stats.norm.ppf(alpha, loc=self.mean_return_daily(), scale=self.sigma_daily())*np.mean(self.adj)

    def cvar(self, alpha):
        def f1(x):
            return x * stats.norm.pdf(x, self.mean_return_daily(), self.sigma_daily())

        def f2(x):
            return stats.norm.pdf(x, self.mean_return_daily(), self.sigma_daily())
        q = stats.norm.ppf(alpha, self.mean_return_daily(), self.sigma_daily())
        # q = self.VaR(alpha=alpha)
        return (integrate.quad(f1, float('-inf'), q)[0] / integrate.quad(f2, float('-inf'), q)[0])*np.mean(self.adj)


pass