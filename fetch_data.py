from pybit.unified_trading import HTTP
import pandas as pd

def fetch_bybit_candles(symbol="BTCUSDT", interval="60", limit=500):
    session = HTTP(testnet=False)
    res = session.get_kline(
        category="spot",
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    df = pd.DataFrame(res['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.astype({
        'open': 'float', 'high': 'float', 'low': 'float',
        'close': 'float', 'volume': 'float', 'turnover': 'float'
    })
    return df