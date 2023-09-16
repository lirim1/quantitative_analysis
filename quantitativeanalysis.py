import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define the ticker symbol for S&P 500
ticker_symbol = "^GSPC"

# Download historical data
sp500_data = yf.download(ticker_symbol, start="2000-01-01", end="2023-01-01")

# Calculate daily returns
sp500_data['Daily_Return'] = sp500_data['Adj Close'].pct_change()

# Calculate moving averages (e.g., 50-day and 200-day)
sp500_data['50_Day_MA'] = sp500_data['Adj Close'].rolling(window=50).mean()
sp500_data['200_Day_MA'] = sp500_data['Adj Close'].rolling(window=200).mean()

# Calculate annualized volatility
annual_volatility = sp500_data['Daily_Return'].std() * np.sqrt(252)

# Calculate the Sharpe ratio (assuming a risk-free rate of 2%)
risk_free_rate = 0.02
sharpe_ratio = (sp500_data['Daily_Return'].mean() - risk_free_rate) / annual_volatility

# Basic statistical analysis
stats_summary = sp500_data['Daily_Return'].describe()

# Visualization
plt.figure(figsize=(12, 6))

# Plot the S&P 500 adjusted close price
plt.subplot(2, 1, 1)
plt.plot(sp500_data['Adj Close'], label='S&P 500')
plt.plot(sp500_data['50_Day_MA'], label='50-Day MA')
plt.plot(sp500_data['200_Day_MA'], label='200-Day MA')
plt.title('S&P 500 Price and Moving Averages')
plt.legend()

# Plot the daily returns distribution
plt.subplot(2, 1, 2)
plt.hist(sp500_data['Daily_Return'].dropna(), bins=50, alpha=0.75)
plt.title('Daily Returns Distribution')

plt.tight_layout()
plt.show()

# Print the results
print("Annualized Volatility: {:.4f}".format(annual_volatility))
print("Sharpe Ratio: {:.4f}".format(sharpe_ratio))
print("\nStatistics Summary:")
print(stats_summary)
