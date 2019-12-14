import pandas as pd
import numpy as np
from datetime import date
import datetime
import matplotlib.pyplot as plt
import yfinance as yf


def datelist(start, end):
    start_date = datetime.date(*start)
    end_date = datetime.date(*end)

    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
    return result


# Create Stocks list
index_list = ['YL', 'SH', 'HL', 'LP', 'ZG', 'HK']
ticker_list = ['600887.SS', '000895.SZ', '002007.SZ', '300003.SZ', '000938.SZ', '002415.SZ']
numstocks = len(ticker_list)
# Prepare Stock Prices
start = date(2018, 11, 1)
end = date(2019, 11, 1)
data = yf.download(ticker_list, start=start, end=end)
StockPrices = data['Adj Close']
StockPrices.columns = index_list


# Return Rate
StockReturns = pd.DataFrame(np.diff(np.log(StockPrices), axis=0))
StockReturns.columns = index_list
LogReturns = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=index_list).append(StockReturns)
# LogReturns.index = datelist(start, end)
stock_return = LogReturns.copy()


# Cumulative Return


def cumulative_return_plot(name_list):
    for name in name_list:
        cumulativereturns = LogReturns[name].cumsum()
        cumulativereturns.plot(label=name)
    plt.legend()
    plt.show()

