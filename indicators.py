import ta

def add_technical_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['volume_change'] = df['volume'].pct_change()
    df['sentiment_score'] = 0.0
    df['funding_rate'] = 0.0
    df['oi_change'] = 0.0
    df['nvt_ratio'] = 0.0
    df['mvrv_zscore'] = 0.0
    return df.dropna()
