import pandas as pd
import numpy as np
import talib as tb
from backtester import BackTester
import os

def prepare_market_data(market_data):
    """
    Enhanced breakout data processing with key supporting indicators.
    """
    dataframe = market_data.copy()
    
    # Calculate daily returns
    dataframe['daily_returns'] = dataframe['close'].pct_change().ffill()
    
    # Breakout levels (multi-timeframe)
    dataframe['breakout_high_25d'] = dataframe['high'].rolling(window=25).max()
    dataframe['breakout_low_25d'] = dataframe['low'].rolling(window=25).min()
    dataframe['highest_high_10'] = dataframe['high'].rolling(window=10).max()
    dataframe['lowest_low_10'] = dataframe['low'].rolling(window=10).min()
    dataframe['highest_high_50'] = dataframe['high'].rolling(window=50).max()
    dataframe['lowest_low_50'] = dataframe['low'].rolling(window=50).min()
    
    # Volatility and risk 
    dataframe['ATR'] = tb.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    dataframe['volatility_20'] = dataframe['daily_returns'].rolling(window=20).std() * np.sqrt(365)
    
    # Trend confirmation
    dataframe['sma_50'] = tb.SMA(dataframe['close'], timeperiod=50)
    dataframe['sma_200'] = tb.SMA(dataframe['close'], timeperiod=200)
    dataframe['ema_20'] = tb.EMA(dataframe['close'], timeperiod=20)
    
    # Volume confirmation
    dataframe['volume_sma_20'] = tb.SMA(dataframe['volume'], timeperiod=20)
    dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma_20']
    
    # Momentum filters
    dataframe['rsi'] = tb.RSI(dataframe['close'], timeperiod=14)
    dataframe['ADX'] = tb.ADX(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
    
    # Breakout strength
    dataframe['distance_from_high'] = (dataframe['close'] - dataframe['breakout_high_25d']) / dataframe['breakout_high_25d']
    dataframe['distance_from_low'] = (dataframe['breakout_low_25d'] - dataframe['close']) / dataframe['breakout_low_25d']
    
    # Fill missing values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.ffill().bfill()
    
    return dataframe

def generate_trading_signals(market_data):
    """
    Enhanced breakout strategy with volume and trend confirmation.
    Core logic: trade breakouts of 25-day highs/lows, with immediate reversals.
    """
    dataframe = market_data.copy()
    
    # Set up signals and state
    dataframe['trade_signals'] = 0
    dataframe['trade_type'] = "HOLD"
    current_position = 0
    stop_loss_price = 0
    
    # Parameters
    base_atr_multiplier = 2.0
    min_volume_ratio = 1.3  # Only breakouts with above-average volume
    
    start_idx = 50  # Wait for all indicators to be available
    
    for i in range(start_idx, len(dataframe)):
        # Skip if any required data is missing
        if (pd.isna(dataframe.iloc[i]['breakout_high_25d']) or pd.isna(dataframe.iloc[i]['breakout_low_25d']) or
            pd.isna(dataframe.iloc[i]['ATR'])):
            continue
        
        # Breakout levels
        prev_high_25 = dataframe.iloc[i-1]['breakout_high_25d']
        prev_low_25 = dataframe.iloc[i-1]['breakout_low_25d']
        
        # Confirmation filters
        volume_confirmed = dataframe.iloc[i]['volume_ratio'] > min_volume_ratio
        long_term_bullish = dataframe.iloc[i]['sma_50'] > dataframe.iloc[i]['sma_200']
        long_term_bearish = dataframe.iloc[i]['sma_50'] < dataframe.iloc[i]['sma_200']
        
        # Volatility-based stop adjustment
        volatility_factor = min(1.8, max(0.8, dataframe.iloc[i]['volatility_20'] / 0.4))
        dynamic_atr_multiplier = base_atr_multiplier * volatility_factor
        
        # Main breakout logic
        if dataframe.iloc[i]['close'] > prev_high_25:
            if current_position == 0 and volume_confirmed:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = 1
                current_position = 1
                dataframe.loc[dataframe.index[i], 'trade_type'] = "LONG_BREAKOUT"
                stop_loss_price = dataframe.iloc[i]['close'] - (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier)
            elif current_position == -1:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = 2
                current_position = 1
                dataframe.loc[dataframe.index[i], 'trade_type'] = "REVERSE_SHORT_TO_LONG"
                stop_loss_price = dataframe.iloc[i]['close'] - (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier)
        
        elif dataframe.iloc[i]['close'] < prev_low_25:
            if current_position == 0 and volume_confirmed:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = -1
                current_position = -1
                dataframe.loc[dataframe.index[i], 'trade_type'] = "SHORT_BREAKOUT"
                stop_loss_price = dataframe.iloc[i]['close'] + (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier)
            elif current_position == 1:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = -2
                current_position = -1
                dataframe.loc[dataframe.index[i], 'trade_type'] = "REVERSE_LONG_TO_SHORT"
                stop_loss_price = dataframe.iloc[i]['close'] + (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier)
        
        # Trailing stop management
        elif current_position == 1:
            stop_loss_price = max(stop_loss_price, 
                              dataframe.iloc[i]['close'] - (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier))
            if dataframe.iloc[i]['close'] < stop_loss_price:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = -1
                current_position = 0
                dataframe.loc[dataframe.index[i], 'trade_type'] = "TRAILING_STOP"
            elif (dataframe.iloc[i]['rsi'] > 80 and dataframe.iloc[i]['ADX'] < 25 and 
                  not long_term_bullish):
                dataframe.loc[dataframe.index[i], 'trade_signals'] = -1
                current_position = 0
                dataframe.loc[dataframe.index[i], 'trade_type'] = "OVERBOUGHT_EXIT"
        
        elif current_position == -1:
            stop_loss_price = min(stop_loss_price, 
                              dataframe.iloc[i]['close'] + (dataframe.iloc[i]['ATR'] * dynamic_atr_multiplier))
            if dataframe.iloc[i]['close'] > stop_loss_price:
                dataframe.loc[dataframe.index[i], 'trade_signals'] = 1
                current_position = 0
                dataframe.loc[dataframe.index[i], 'trade_type'] = "TRAILING_STOP"
            elif (dataframe.iloc[i]['rsi'] < 20 and dataframe.iloc[i]['ADX'] < 25 and 
                  not long_term_bearish):
                dataframe.loc[dataframe.index[i], 'trade_signals'] = 1
                current_position = 0
                dataframe.loc[dataframe.index[i], 'trade_type'] = "OVERSOLD_EXIT"
    
    return dataframe

def main():
    """
    Main function to run the strategy and evaluate its performance.
    """
    print("Loading data...")
    data = pd.read_csv("BTC_2019_2023_1d.csv")
    
    print("Processing data and calculating indicators...")
    processed_data = prepare_market_data(data)
    
    print("Generating trading signals...")
    result_data = generate_trading_signals(processed_data)
    
    print("Saving results to CSV...")
    csv_file_path = "final_data.csv" 
    result_data.to_csv(csv_file_path, index=False)
    
    print("Running backtester...")
    bt = BackTester("BTC", signal_data_path="final_data.csv", master_file_path="final_data.csv", compound_flag=1)
    bt.get_trades(1000)
    
    # Print trades and their PnL
    print("\nTrade Details:")
    print("=" * 50)
    for trade in bt.trades:
        print(trade)
        print(f"PnL: ${trade.pnl():.2f}")
        print("-" * 50)
    
    # Print results
    print("\nPerformance Statistics:")
    print("=" * 50)
    stats = bt.get_statistics()
    for key, val in stats.items():
        print(f"{key}: {val}")
    
    # Check for lookahead bias (sample only a few signals for speed)
    print("\nChecking for lookahead bias (sampling)...")
    signal_indices = [i for i in range(len(result_data)) if result_data.iloc[i]['trade_signals'] != 0]
    
    # Sample at most 5 signals for checking
    sample_size = min(5, len(signal_indices))
    if sample_size > 0:
        sample_indices = np.random.choice(signal_indices, sample_size, replace=False)
        
        lookahead_bias = False
        for i in sample_indices:
            print(f"Checking signal at index {i}")
            temp_data = data.iloc[:i+1].copy()
            temp_data = prepare_market_data(temp_data)
            temp_data = generate_trading_signals(temp_data)
            if temp_data.iloc[i]['trade_signals'] != result_data.iloc[i]['trade_signals']:
                print(f"Lookahead bias detected at index {i}")
                lookahead_bias = True
        
        if not lookahead_bias:
            print("No lookahead bias detected in sampled signals.")
    else:
        print("No signals to check for lookahead bias.")
    
    # Generate the PnL graph
    print("\nGenerating trade and PnL graphs...")
    bt.make_trade_graph()
    bt.make_pnl_graph()
    
    print("\nBacktesting complete!")
    
if __name__ == "__main__":
    main()
