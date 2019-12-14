from Portfolio import stock_return, LogReturns, numstocks
import numpy as np

# Statistic Weights
portfolio_weights = np.array([0.2, 0.18, 0.12, 0.13, 0.17, 0.2])
WeightedReturn = stock_return.mul(portfolio_weights, axis=1)
LogReturns['Portfolio'] = WeightedReturn.sum(axis=1)
# StockReturns.Portfolio.plot()
# plt.show()

# Equal Weighted Portfolio
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)
LogReturns['Portfolio_EW'] = stock_return.mul(portfolio_weights_ew, axis=1).sum(axis=1)

# Asset Value Weighted Portfolio
market_capitals = np.array([179.682, 106.899, 51.591, 56.475, 57.851, 313.903])
mcap_weights = market_capitals / sum(market_capitals)
LogReturns['Portfolio_MCAP'] = stock_return.mul(mcap_weights, axis=1).sum(axis=1)