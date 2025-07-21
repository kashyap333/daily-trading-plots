from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def load_price_data(
    start_date='2020-01-01',
    end_date=datetime.today(),
    assets=["Stock", "Bond", "Commodity"],  # or use "All" for everything
    path_stock="D:\\Quant\\Datas\\master_stock_data.csv",
    path_bond="D:\\Quant\\Datas\\master_bond_etf_data.csv",
    path_commodity="D:\\Quant\\Datas\\master_commodity_etf_data.csv"
):
    if end_date is None:
        end_date = datetime.now()

    # Convert to Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    def load_and_filter(filepath, asset_type):
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df[(df['Date'] > start_date) & (df['Date'] <= end_date)]
        df['AssetType'] = asset_type
        return df

    # Normalize asset input
    if isinstance(assets, str):
        assets = [assets]
    if "All" in assets:
        assets = ["Stock", "Bond", "Commodity"]

    combined = []

    if "Stock" in assets:
        combined.append(load_and_filter(path_stock, "Stock"))

    if "Bond" in assets:
        try:
            combined.append(load_and_filter(path_bond, "Bond"))
        except FileNotFoundError:
            print(f"Warning: {path_bond} not found. Skipping Bond data.")

    if "Commodity" in assets:
        try:
            combined.append(load_and_filter(path_commodity, "Commodity"))
        except FileNotFoundError:
            print(f"Warning: {path_commodity} not found. Skipping Commodity data.")

    if not combined:
        raise ValueError("No valid asset types selected or no files found.")

    final_df = pd.concat(combined, ignore_index=True)
    final_df.sort_values(by="Date", inplace=True)

    return final_df


def filter_by_var(price_df, confidence_level=0.95, var_threshold=-0.05, lookback=252, method='historical'):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()
    var_series = {}
    for symbol, r in returns.groupby('Symbol'):
        r = r.iloc[-lookback:]
        if len(r) == 0:
            continue
        if method == 'historical':
            var = np.percentile(r, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mu = r.mean()
            sigma = r.std()
            z = norm.ppf(1 - confidence_level)
            var = mu + z * sigma
        else:
            raise ValueError("Method must be 'historical' or 'parametric'.")
        var_series[symbol] = var
    var_df = pd.Series(var_series)
    return var_df[var_df >= var_threshold].index.tolist()


def filter_by_volatility(price_df, window=20, min_vol=0.005, max_vol=0.05):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()
    rolling_vol = returns.groupby('Symbol').rolling(window).std().reset_index(level=0, drop=True)
    last_vol = rolling_vol.groupby(returns.index.get_level_values(0)).last() if isinstance(rolling_vol.index, pd.MultiIndex) else rolling_vol.groupby('Symbol').last()
    filtered = last_vol[(last_vol >= min_vol) & (last_vol <= max_vol)]
    return filtered.index.tolist()


def filter_by_trend(price_df, window=60, min_return=0.0):
    returns = price_df.groupby('Symbol')['Close'].apply(lambda x: x.pct_change()).dropna()

    # Convert to wide format to easily compute across time
    returns_df = returns.unstack(level='Symbol')
    
    # Rolling cumulative returns
    cum_returns = (1 + returns_df).rolling(window=window).apply(np.prod, raw=True) - 1

    # Get latest returns
    latest_returns = cum_returns.iloc[-1]
    
    selected = latest_returns[latest_returns > min_return].index.tolist()
    return selected


def filter_by_correlation(price_df, corr_threshold=0.3):
    # Make sure 'Date' is a column (not index)
    df = price_df.reset_index()

    # Pivot to wide format (Date x Symbols) for correlation calculation
    returns = df.pivot(index='Date', columns='Symbol', values='Close').pct_change().dropna()
    
    corr_matrix = returns.corr()

    selected = []
    for asset in corr_matrix.columns:
        if all(abs(corr_matrix.loc[asset, other]) < corr_threshold for other in selected):
            selected.append(asset)

    return selected, corr_matrix.loc[selected, selected]


def select_assets_by_sharpe(price_df, risk_free_rate=0.0, top_n=None, min_sharpe=None):
    df_wide = price_df.pivot(index='Date', columns='Symbol', values='Close')
    returns = df_wide.pct_change().dropna()
    rf_daily = (1 + risk_free_rate)**(1/252) - 1

    sharpe_data = []
    for asset in returns.columns:
        mean_ret = returns[asset].mean()
        std_dev = returns[asset].std()
        if std_dev == 0 or np.isnan(std_dev):
            sharpe = np.nan
        else:
            sharpe = (mean_ret - rf_daily) / std_dev
        sharpe_data.append({
            'asset': asset,
            'mean_return': mean_ret,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe
        })

    sharpe_df = pd.DataFrame(sharpe_data).dropna().set_index('asset').sort_values('sharpe_ratio', ascending=False)

    if top_n is not None:
        selected_assets = sharpe_df.head(top_n).index.tolist()
    elif min_sharpe is not None:
        selected_assets = sharpe_df[sharpe_df['sharpe_ratio'] >= min_sharpe].index.tolist()
    else:
        selected_assets = sharpe_df.index.tolist()

    return sharpe_df, selected_assets

def simple_moving_average(price_df, short_window=50, long_window=200):
    """
    Compute SMA signals avoiding lookahead bias (uses previous day's moving averages).
    Returns wide-format signal DataFrame.

    Args:
        price_df (pd.DataFrame): Must contain 'Date', 'Symbol', 'Close'.
        short_window (int): Window for short SMA.
        long_window (int): Window for long SMA.

    Returns:
        sma_short (pd.DataFrame): Short window SMA (shifted by 1 day).
        sma_long (pd.DataFrame): Long window SMA (shifted by 1 day).
        signal_df (pd.DataFrame): Signals: 1 for long, -1 for short, 0 for neutral.
    """

    # Check required columns
    required_cols = {'Date', 'Symbol', 'Close'}
    if not required_cols.issubset(price_df.columns):
        raise ValueError(f"Input DataFrame must contain {required_cols}")

    # Sort by Date and Symbol
    price_df_sorted = price_df.sort_values(by=['Date', 'Symbol']).copy()

    # Pivot to wide format with Date index and Symbol columns
    prices = price_df_sorted.pivot(index='Date', columns='Symbol', values='Close')

    # Calculate rolling SMAs and shift by 1 day to avoid lookahead bias
    sma_short = prices.rolling(window=short_window).mean().shift(1)
    sma_long = prices.rolling(window=long_window).mean().shift(1)

    # Initialize signals with 0
    signal_df = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Generate signals
    signal_df[sma_short > sma_long] = 1
    signal_df[sma_short < sma_long] = -1
    signal_df = signal_df.ffill().fillna(0)

    
    return signal_df

def ewma_momentum_signals(price_df, span=60, threshold=0.001, min_days_above_thresh=5):
    df = price_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'Symbol'])
    df = df.set_index('Date')

    prices = df.pivot(columns="Symbol", values="Close")

    shifted_prices = prices.shift(1)
    log_returns = np.log(prices / shifted_prices)

    momentum_df = log_returns.ewm(span=span, adjust=False).mean()

    pos_momentum = (momentum_df > threshold).astype(int)
    neg_momentum = (momentum_df < -threshold).astype(int)

    pos_count = pos_momentum.rolling(window=span).sum()
    neg_count = neg_momentum.rolling(window=span).sum()

    long_signal = (pos_count >= min_days_above_thresh).astype(int)
    short_signal = (neg_count >= min_days_above_thresh).astype(int)

    # Clean signals: if both long and short active, set to 0 (neutral)
    signal_df = long_signal - short_signal
    conflict_mask = (long_signal == 1) & (short_signal == 1)
    signal_df[conflict_mask] = 0

    return momentum_df, signal_df, long_signal


def inverse_volatility_weights(df, lookback=60, price_column="Close"):
    # Reset index so Date becomes a column for pivoting
    df = df.reset_index()
    
    price_df = df.pivot(index='Date', columns='Symbol', values=price_column)
    returns = price_df.pct_change()
    rolling_vol = returns.rolling(window=lookback).std().shift(1)  # Shift to avoid lookahead bias
    inv_vol = 1 / rolling_vol
    weights_df = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights_df


def compute_combined_weights_for_date(price_df, date):
    # Ensure date and index are datetime
    date = pd.to_datetime(date)
    price_df.index = pd.to_datetime(price_df.index)
    
    # Use data up to 'date'
    df = price_df[price_df.index <= date]
    
    # Step 1: Filter by VaR (risk)
    safe_assets = filter_by_var(price_df)
    df = df[df['Symbol'].isin(safe_assets)]
    
    # Step 2: Filter by volatility
    stable_assets = filter_by_volatility(price_df=df)
    df = df[df['Symbol'].isin(stable_assets)]
    
    # Step 3: Filter by correlation
    final_assets, _ = filter_by_correlation(df, corr_threshold=0.9)
    df = df[df['Symbol'].isin(final_assets)]
    
    # Step 4: Compute momentum signals
    #signals = simple_moving_average(price_df, short_window=10, long_window=70)
    momentum_df, signals1, signals = ewma_momentum_signals(price_df, span=60, threshold=0.003, min_days_above_thresh=10)
    
    # Step 5: Compute rolling max sharpe weights (returns DataFrame for all dates)
    weights = inverse_volatility_weights(df)
    
    # Align weights and signals on dates and assets
    common_dates = signals.index.intersection(weights.index)
    common_assets = signals.columns.intersection(weights.columns)
     
    signals_aligned = signals.loc[common_dates, common_assets]
    weights_aligned = weights.loc[common_dates, common_assets]
    
    # Calculate combined weights = signals * weights
    combined_weights = signals_aligned * weights_aligned
    
    # Normalize weights per date (sum absolute weights = 1)
    abs_sum = combined_weights.abs().sum(axis=1).replace(0, np.nan)
    combined_weights = combined_weights.div(abs_sum, axis=0).fillna(0)
    
    return combined_weights, signals
    

def backtest_close_to_close(price_df, weights_df, allow_short=False):
    """
    Backtest portfolio returns from close-to-close prices and given weights.

    Args:
        price_df (pd.DataFrame): Long format with 'Date' index, 'Symbol' column, and 'Close' prices.
        weights_df (pd.DataFrame): Wide format with Date index, Symbols as columns; values are weights.
        allow_short (bool): If False, clips negative weights to zero (no shorting).

    Returns:
        pd.Series: Daily portfolio returns indexed by Date.
    """
    # Filter price data to symbols in weights
    symbols = weights_df.columns
    price_df = price_df[price_df['Symbol'].isin(symbols)].copy()

    # Set Date index if not already
    if price_df.index.name != 'Date':
        price_df = price_df.set_index('Date')
    price_df = price_df.sort_index()
    price_df = price_df.sort_values(['Date', 'Symbol'])

    # Dates to consider: intersection of price and weights dates
    dates = sorted(set(price_df.index.unique()) & set(weights_df.index))

    portfolio_returns = []
    portfolio_dates = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        if prev_date not in weights_df.index:
            continue

        weights = weights_df.loc[prev_date].copy()
        if not allow_short:
            weights = weights.clip(lower=0)

        try:
            close_prev = price_df.loc[prev_date].set_index('Symbol')['Close']
            close_curr = price_df.loc[curr_date].set_index('Symbol')['Close']
        except KeyError:
            # If price data missing, skip that day
            continue

        # Calculate asset returns
        asset_returns = (close_curr / close_prev - 1).reindex(weights.index).fillna(0)

        # Portfolio return = sum(weights * returns)
        port_return = (weights * asset_returns).sum()

        portfolio_dates.append(curr_date)
        portfolio_returns.append(port_return)

    return pd.Series(portfolio_returns, index=portfolio_dates).sort_index()


def backtest_metrics_close_to_close(price_df, combined_weights, freq=252, allow_short = False):
    returns = backtest_close_to_close(price_df, combined_weights, allow_short = allow_short)

    cumulative_return = (1 + returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    volatility = returns.std() * np.sqrt(freq)
    sharpe = annualized_return / volatility if volatility != 0 else np.nan
    metrics = {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
    }
    return returns, metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def save_performance_with_weights_plot(portfolio_returns, weights_df, filename='performance_plot.png',
                                       allow_short=True, min_periods=21, dpi=300):
    """
    Save a plot of cumulative return, Sharpe, drawdown, and color the performance line 
    based on the sign of total weights (long/short/neutral).

    Args:
        portfolio_returns (pd.Series): Daily portfolio returns indexed by Date.
        weights_df (pd.DataFrame): Wide-format DataFrame with asset weights per day.
        filename (str): Path to save the plot image (e.g., 'output.png').
        allow_short (bool): If False, all negative weights are treated as zero.
        min_periods (int): Minimum observations to compute Sharpe.
        dpi (int): Resolution of saved image.
    """

    # Align weights to returns
    weights_aligned = weights_df.reindex(portfolio_returns.index)

    if not allow_short:
        weights_aligned = weights_aligned.clip(lower=0)

    # Determine position state from weights
    def determine_position(weights_row):
        if weights_row.abs().sum() == 0:
            return 0  # Neutral
        elif weights_row.sum() > 0:
            return 1  # Net long
        elif weights_row.sum() < 0:
            return -1  # Net short
        else:
            return 0

    position_state = weights_aligned.apply(determine_position, axis=1).shift(1).fillna(0).astype(int)

    # Cumulative returns and drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1

    # Sharpe Ratio
    cum_mean = portfolio_returns.expanding(min_periods=min_periods).mean()
    cum_std = portfolio_returns.expanding(min_periods=min_periods).std().clip(lower=1e-6)
    cumulative_sharpe = (cum_mean / cum_std) * np.sqrt(252)
    cumulative_sharpe[:min_periods] = np.nan

    # === Plotting ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    colors = {1: 'green', -1: 'red', 0: 'gray'}
    last_pos = None
    segment_start = None

    for date, pos in position_state.items():
        if last_pos is None:
            last_pos = pos
            segment_start = date
            continue
        if pos != last_pos:
            segment = cumulative_returns.loc[segment_start:date]
            ax1.plot(segment.index, segment.values - 1, color=colors.get(last_pos, 'gray'), linewidth=2)
            segment_start = date
            last_pos = pos

    if segment_start is not None and not cumulative_returns.empty:
        segment = cumulative_returns.loc[segment_start:]
        ax1.plot(segment.index, segment.values - 1, color=colors.get(last_pos, 'gray'), linewidth=2)

    # Sharpe on secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(cumulative_sharpe.index, cumulative_sharpe, label="Cumulative Sharpe", color='blue', alpha=0.6)
    ax1.set_title("Cumulative Return Colored by Position (Based on Weights)")
    ax1.set_ylabel("Return")
    ax1b.set_ylabel("Sharpe Ratio")

    # Drawdown
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Drawdown")

    # Legend
    legend_lines = [
        mlines.Line2D([], [], color='green', linewidth=2, label='Net Long (> 0)'),
        mlines.Line2D([], [], color='gray', linewidth=2, label='Neutral (0)')
    ]
    if allow_short:
        legend_lines.append(mlines.Line2D([], [], color='red', linewidth=2, label='Net Short (< 0)'))

    ax1.legend(handles=legend_lines, loc='upper left')

    fig.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)  # Close the figure to free memory


df = load_price_data(
    start_date='2020-01-01',
    end_date=datetime.today(),
    assets=["Stock"],
)

combined_weights, signals = compute_combined_weights_for_date(df, '2025-01-01')

returns, metrics = backtest_metrics_close_to_close(df, combined_weights, allow_short=True)

filename="D:\\Quant\\daily-trading-plots\\plot.png"

save_performance_with_weights_plot(
    portfolio_returns=returns,
    weights_df=combined_weights,
    filename=filename,
    allow_short=False
)

import subprocess
from datetime import datetime

repo_path = "D:\\Quant\\daily-trading-plots"
plot_path = os.path.join(repo_path, "plot.png")

# Ensure file exists
if not os.path.exists(plot_path):
    raise FileNotFoundError(f"{plot_path} not found!")

# Stage only the plot (forcefully in case it's ignored)
subprocess.run(["git", "add", "-f", plot_path], cwd=repo_path, check=True)

# Optional: also add any modified code (like daily_results.py)
subprocess.run(["git", "add", "."], cwd=repo_path, check=True)


subprocess.run(["git", "commit", "-m", "Add plot"], cwd=repo_path, check=True)
subprocess.run(["git", "push"], cwd=repo_path, check=True)
