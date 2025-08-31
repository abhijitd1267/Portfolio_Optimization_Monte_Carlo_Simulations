# Portfolio Optimization & Monte Carlo Simulations

## Overview

This project analyzes stock portfolios using historical data from 2018 onward. It covers data collection, return/risk calculations, Sharpe Ratio optimization, and Monte Carlo price simulations.

- **Stocks:** AMZN, JPM, META, PG, GOOGL, CAT, PFE, EXC, DE, JNJ  
- **Techniques:** Efficient Frontier, portfolio optimization (SciPy), correlated simulations (Cholesky).  
- **Dataset:** Fetched via yfinance; saved as stock_data.csv.  
- **Example Output:** Optimized Sharpe Ratio ~1.5.  

See [Portfolio Optimization.ipynb](Portfolio%20Optimization.ipynb) for code.

## Table of Contents

1. [Mini Project 1: Collect & Inspect Data](#mini-project-1-collect--inspect-data)  
2. [Mini Project 2: Portfolio Return & Risk](#mini-project-2-portfolio-return--risk)  
3. [Mini Project 3: Portfolio Optimization](#mini-project-3-portfolio-optimization)  
4. [Mini Project 4: Monte Carlo Simulations](#mini-project-4-monte-carlo-simulations)  

## Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
import yfinance as yf
from scipy.optimize import minimize
```

## Mini Project 1: Collect & Inspect Data

Fetch adjusted closes and compute daily returns.

```python
stocks = ['AMZN', 'JPM', 'META', 'PG', 'GOOGL', 'CAT', 'PFE', 'EXC', 'DE', 'JNJ']
start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()
stock_data = yf.download(stocks, start=start, end=end)['Adj Close']
daily_returns = stock_data.pct_change().dropna()
```

- **Inspection:** `.head()`, `.describe()`.  
- **Plot:** Interactive price lines via Plotly.

## Mini Project 2: Portfolio Return & Risk

Equal-weight portfolio metrics.

```python
weights = np.array([1/len(stocks)] * len(stocks))
portfolio_return = np.dot(weights, daily_returns.mean()) * 252
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
sharpe = (portfolio_return - 0.02) / portfolio_vol
```

- **Efficient Frontier:** 10,000 random portfolios plotted (return vs. vol, colored by Sharpe).

## Mini Project 3: Portfolio Optimization

Maximize Sharpe Ratio.

```python
def neg_sharpe(weights):
    ret = np.dot(weights, daily_returns.mean()) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    return - (ret - 0.02) / vol

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in stocks]
result = minimize(neg_sharpe, weights, method='SLSQP', bounds=bounds, constraints=constraints)
```

- **Results:** Optimal weights and Sharpe.  
- **Plot:** Frontier with optimal point.

## Mini Project 4: Monte Carlo Simulations

Forecast prices with Brownian motion.

```python
# For AMZN (extend for portfolio)
mu, sigma = daily_returns['AMZN'].mean(), daily_returns['AMZN'].std()
start_price = stock_data['AMZN'].iloc[-1]
num_sims, num_days = 1000, 252
sims = np.full([num_days, num_sims], start_price)
for t in range(1, num_days):
    sims[t] = sims[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal(0, 1, num_sims))
```

- **Correlations:** Use Cholesky on cov matrix.  
- **Plot:** Simulated paths (Plotly line).

## Requirements

- Python 3.x  
- Libraries: pandas, numpy, matplotlib, plotly, yfinance, scipy  

```bash
pip install pandas numpy matplotlib plotly yfinance scipy
```
