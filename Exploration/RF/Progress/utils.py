'''
# Features

### Basic Price Features
Percentage change in price [Capture short term price movement and direction]
Log of price rations       [Symmetric and stable, may prove useful]

### Volatility Features
Rolling standard deviation of returns over  5 and 10 days [Shows risk factors clearly]

### Volume Features
Percentage change in trading volume [To indicate price movement and seperation from volatility]
5 day average volume                [To identify volume spikes]
Volume relative to average          [Normalizes volume across time](Consider removing this, might lead to unnessary trees)

### Simple Moving Average
5, 10, 20 day moving averages [smoothes the noise in the data to uncover underlying trends](Have several for different trends)

### Price Relative to Moving Average
Price divided by moving average [Normalize position relative to trend]
value > 1 means above average (overbought, expect to drop)
value < 1 means below average (oversold, expect to rise)

### Moving Average Crossover
difference between 5 day and 10 day SMA  [Golden/death cross]
diff between 10 day and 20 day SMA

Golden death cross is a signal that indicates prediction of bullish or bearish future trends
Positive = short term above long term (Bullish)
Negative = short temr under long term (Bearish)
this might be something that the Random Forest might need more guidance to properly use

There are also some other key trends that depict markets, I can't remember rn, but there is one specific one that has no mathematical expectation, but returns are very favorable for long term investments (!TODO)

### Momentum Features
Price Change over 5 and 10 days [Measure upward and downward trend strength]

### Price Range Features
Daily price range (Highest value - Lowest Value before close)
Position of high candle when closed   
Position of low candle when closed 

if close_to_high is near 1 [Means strong buying pressure](Might be kinda redundant considering we have SMA, conisder removing SMA)
if close_to_low is near 1  [Means strong selling pressure]
'''







import numpy as np
import pandas as pd



# Feature Engineering

def create_features(df):
    df = df.copy()


    # ATR 
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()



    # moving averages & ratios 
    '''
    NOTE Remember to shift to prevent lookahead bias
    '''
    # EMA is similiar to SMA but more weight is given to recent prices (Can be good for volatility)
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()

    df['sma_5'] = df['Close'].rolling(5).mean().shift()
    df['sma_10'] = df['Close'].rolling(10).mean().shift()
    df['sma_15'] = df['Close'].rolling(15).mean().shift()  # Added for 15-day horizon
    df['sma_20'] = df['Close'].rolling(20).mean().shift()
    df['sma_30'] = df['Close'].rolling(30).mean().shift()
    df['sma_50'] = df['Close'].rolling(50).mean().shift()  # Added longer-term trend

    # MACD 
    '''
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. 
    The MACD (HERE) is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.
    '''
    EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD_12_26'] = pd.Series(EMA_12 - EMA_26)
    # NOTE Named it signal_9 even though its using 12_26 to stress that we are using another EMA of 9 to act as a signal line (refernce)
    df['MACD_signal_9'] = pd.Series(df.MACD_12_26.ewm(span=9, min_periods=9).mean())


    df['price_to_sma_5'] = df['Close'] / df['sma_5']
    df['price_to_sma_10'] = df['Close'] / df['sma_10']
    df['price_to_sma_15'] = df['Close'] / df['sma_15']  # Added for 15-day horizon
    df['price_to_sma_20'] = df['Close'] / df['sma_20']
    df['price_to_sma_50'] = df['Close'] / df['sma_50']  # Added longer-term position

    # Explore this more, it is very useful for volatility and trend analysis (I am experimenting with switching target variable to log returns)
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [2,3,5,10,20]:
        df[f'log_return_{lag}'] = np.log(df['Close'] / df['Close'].shift(lag))

    # volatility 
    df['realized_vol_5'] = df['log_return_1'].rolling(5).apply(lambda x: np.sqrt(np.mean(x**2)))
    df['realized_vol_15'] = df['log_return_1'].rolling(15).apply(lambda x: np.sqrt(np.mean(x**2)))  # Added for 15-day horizon
    df['realized_vol_20'] = df['log_return_1'].rolling(20).apply(lambda x: np.sqrt(np.mean(x**2)))
    df['realized_vol_30'] = df['log_return_1'].rolling(30).apply(lambda x: np.sqrt(np.mean(x**2)))  # Added longer-term volatility
    df['vol_ratio_5_20'] = df['realized_vol_5'] / df['realized_vol_20']
    df['vol_ratio_15_30'] = df['realized_vol_15'] / df['realized_vol_30']  # Added for 15-day horizon


    # SMA slopes 
    def slope(series):
        y = series.values
        x = np.arange(len(y))
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    df['sma_slope_5'] = df['Close'].rolling(5).apply(slope, raw=False)
    df['sma_slope_10'] = df['Close'].rolling(10).apply(slope, raw=False)
    df['sma_slope_15'] = df['Close'].rolling(15).apply(slope, raw=False)  # Added for 15-day horizon
    df['sma_slope_20'] = df['Close'].rolling(20).apply(slope, raw=False)
    df['sma_slope_30'] = df['Close'].rolling(30).apply(slope, raw=False)  # Added longer-term trend

    # momentum
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    df['momentum_15'] = df['Close'].pct_change(15)  # Added for 15-day horizon - key feature!
    df['momentum_30'] = df['Close'].pct_change(30)  # Added longer-term momentum

    # RSI 
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + RS))

    # Stochastics 
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # volume 
    df['Volume_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_roc'] = df['OBV'].pct_change()

    # candlesticks 
    df['real_body'] = (df['Close'] - df['Open']).abs()
    df['upper_shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['body_to_range'] = df['real_body'] / (df['High'] - df['Low'])

    # misc 
    df['High_Low_range'] = df['High'] - df['Low']

    df['return_mean_5']  = df['log_return_1'].rolling(5).mean()
    df['return_std_5']   = df['log_return_1'].rolling(5).std()
    df['return_std_10']  = df['log_return_1'].rolling(10).std()

    df['volume_change_1'] = df['Volume'].pct_change()
    df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    df['daily_range'] = (df['High'] - df['Low']) / df['Open']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)




    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

