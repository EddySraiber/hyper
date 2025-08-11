#!/usr/bin/env python3
"""
Trading Performance Analysis Script
"""
import json
import statistics

# Load trade outcomes
with open('/home/eddy/Hyper/data/trade_outcomes.json', 'r') as f:
    trades = json.load(f)

# Calculate key metrics
total_trades = len(trades)
wins = sum(1 for t in trades if t['result'] == 'win')
losses = sum(1 for t in trades if t['result'] == 'loss')
breakevens = sum(1 for t in trades if t['result'] == 'breakeven')

win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
loss_rate = (losses / total_trades) * 100 if total_trades > 0 else 0

# PnL analysis
pnl_values = [t['pnl_absolute'] for t in trades]
pnl_percentages = [t['pnl_percentage'] for t in trades]

total_pnl = sum(pnl_values)
avg_pnl = statistics.mean(pnl_values) if pnl_values else 0
median_pnl = statistics.median(pnl_values) if pnl_values else 0
pnl_std = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0

avg_pnl_pct = statistics.mean(pnl_percentages) if pnl_percentages else 0
median_pnl_pct = statistics.median(pnl_percentages) if pnl_percentages else 0

# Winning vs losing trade analysis
winning_trades = [t['pnl_absolute'] for t in trades if t['result'] == 'win']
losing_trades = [t['pnl_absolute'] for t in trades if t['result'] == 'loss']

avg_win = statistics.mean(winning_trades) if winning_trades else 0
avg_loss = statistics.mean(losing_trades) if losing_trades else 0

sum_wins = sum(winning_trades) if winning_trades else 0
sum_losses = abs(sum(losing_trades)) if losing_trades else 0
profit_factor = sum_wins / sum_losses if sum_losses != 0 else float('inf')

# Duration analysis
durations = [t['duration_minutes'] for t in trades if 'duration_minutes' in t]
avg_duration = statistics.mean(durations) if durations else 0

# News accuracy analysis
accuracies = [t['news_accuracy'] for t in trades if 'news_accuracy' in t and t['news_accuracy'] is not None]
avg_news_accuracy = statistics.mean(accuracies) * 100 if accuracies else 0

# Decision quality analysis
decision_qualities = [t['decision_quality'] for t in trades if 'decision_quality' in t and t['decision_quality'] is not None]
avg_decision_quality = statistics.mean(decision_qualities) * 100 if decision_qualities else 0

# Calculate Sharpe ratio (corrected implementation)
def calculate_sharpe_ratio(pnl_values, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate Sharpe ratio correctly.
    Sharpe = (Mean Return - Risk Free Rate) / Standard Deviation of Returns
    Annualized using square root of time.
    """
    if len(pnl_values) < 2:
        return 0.0
    
    # Convert absolute P&L to returns (percentage of initial capital)
    initial_capital = 100000
    returns = [pnl / initial_capital for pnl in pnl_values]
    
    if statistics.stdev(returns) == 0:
        return 0.0
    
    # Calculate daily risk-free rate
    daily_risk_free = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = [r - daily_risk_free for r in returns]
    
    # Calculate Sharpe ratio and annualize
    mean_excess = statistics.mean(excess_returns)
    std_returns = statistics.stdev(returns)
    
    # Annualize the Sharpe ratio (assuming trades are roughly daily frequency)
    # Note: This is approximate since trade frequency varies
    sharpe_ratio = (mean_excess / std_returns) * (periods_per_year ** 0.5) if std_returns != 0 else 0
    
    return sharpe_ratio

sharpe_ratio = calculate_sharpe_ratio(pnl_values)

# Maximum drawdown calculation (corrected implementation)
def calculate_max_drawdown(pnl_values, initial_capital=100000):
    """
    Calculate maximum drawdown correctly.
    Drawdown = (Peak Value - Trough Value) / Peak Value
    Maximum drawdown is the largest such drawdown over the period.
    """
    if not pnl_values:
        return 0.0
    
    # Calculate cumulative P&L starting from initial capital
    cumulative_pnl = [initial_capital]
    running_total = initial_capital
    
    for pnl in pnl_values:
        running_total += pnl
        cumulative_pnl.append(running_total)
    
    # Calculate maximum drawdown
    max_dd = 0.0
    peak = cumulative_pnl[0]
    
    for value in cumulative_pnl:
        # Update peak if we have a new high
        if value > peak:
            peak = value
        
        # Calculate drawdown from peak
        if peak > 0:
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
    
    return max_dd

max_dd = calculate_max_drawdown(pnl_values)

print('=== TRADING SYSTEM PERFORMANCE ANALYSIS ===')
print(f'Total Trades: {total_trades}')
print(f'Wins: {wins} ({win_rate:.1f}%)')
print(f'Losses: {losses} ({loss_rate:.1f}%)')  
print(f'Breakevens: {breakevens} ({(breakevens/total_trades)*100:.1f}%)')
print(f'')
print(f'=== P&L ANALYSIS ===')
print(f'Total P&L: ${total_pnl:.2f}')
print(f'Average P&L per trade: ${avg_pnl:.2f}')
print(f'Median P&L per trade: ${median_pnl:.2f}')
print(f'P&L Standard Deviation: ${pnl_std:.2f}')
print(f'Average P&L %: {avg_pnl_pct:.2f}%')
print(f'Median P&L %: {median_pnl_pct:.2f}%')
print(f'')
print(f'=== WIN/LOSS ANALYSIS ===')
print(f'Average Winning Trade: ${avg_win:.2f}')
print(f'Average Losing Trade: ${avg_loss:.2f}')
print(f'Profit Factor: {profit_factor:.2f}')
print(f'')
print(f'=== RISK METRICS ===')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
print(f'Maximum Drawdown: {max_dd*100:.1f}%')
print(f'')
print(f'=== TIMING ANALYSIS ===')
print(f'Average Trade Duration: {avg_duration:.0f} minutes ({avg_duration/60:.1f} hours)')
print(f'')
print(f'=== QUALITY METRICS ===')
print(f'Average News Accuracy: {avg_news_accuracy:.1f}%')
print(f'Average Decision Quality: {avg_decision_quality:.1f}%')