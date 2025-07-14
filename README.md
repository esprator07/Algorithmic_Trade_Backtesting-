# Algorithmic_Trade_Backtesting-
This project provides a sophisticated backtesting environment for three distinct exit strategies based on the Relative Strength Index (RSI) indicator, designed for trading on the Binance Futures market. The scripts are written in Python and leverage the python-binance and TA-Lib libraries.
Core Strategy Logic
All strategies share a common entry signal:
* LONG Signal: Triggered when the RSI crosses back above the overbought level after previously dipping below it.
* SHORT Signal: Triggered when the RSI crosses back below the oversold level after previously rising above it.
The primary goal of this project is to compare the performance of this standard entry signal when paired with different exit mechanisms.

The Three Exit Strategies Tested
This repository contains separate scripts for the following three strategies:
1. Fixed Take Profit / Stop Loss (TP/SL): The most fundamental and disciplined approach. A position is closed upon reaching a predefined, static Take Profit or Stop Loss percentage. The risk-reward ratio is fixed.
2. Trailing Stop Loss (TSL): Focuses on following the trend. There is no fixed profit target. The stop-loss level dynamically updates as the price moves favorably, trailing the new peak/trough. The goal is to maximize profit when a strong trend is captured.
3. Hybrid Model (TP + TSL): A combination of the two. It uses both a fixed Take Profit target and a Trailing Stop Loss simultaneously. It aims to secure guaranteed profits during sharp price movements while maximizing gains during slower, more sustained trends.

Key Features
High-Precision Backtesting: All entry and exit points are determined with high precision by scanning 5-minute data within the main 4-hour chart, increasing the realism of the backtest results.
Realistic Cost Calculation: Each backtest accounts for trading commissions and accumulated funding fees, providing a net profit/loss report.
Comprehensive Optimization: The scripts include an optimization manager to automatically test various combinations of critical parameters like RSI, TP, SL, and TSL.
Detailed Reporting: At the end of each test, a detailed report is generated and saved to a file. It includes metrics such as overall PNL, win rate, total number of trades, performance by position type (long/short), and per-coin profitability.
This tool is designed for traders and developers who want to conduct an in-depth analysis of how RSI-based mean-reversion strategies perform with different risk management approaches.
