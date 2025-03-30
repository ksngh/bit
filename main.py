from fetch_data import fetch_bybit_candles
from indicators import add_technical_indicators
from model import preprocess_data, create_lstm_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

features = ['rsi', 'macd', 'volume_change', 'obv', 'funding_rate', 'oi_change', 'nvt_ratio', 'mvrv_zscore', 'sentiment_score']
target = 'close'

# (1) 데이터 수집 및 전처리
df = fetch_bybit_candles()
df = add_technical_indicators(df)
X, y, scaler = preprocess_data(df, features, target)

# (2) 모델 구성 및 학습
model = create_lstm_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# (3) 예측 및 시각화
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(
    [[*([0]*(len(features))), p[0]] for p in predicted]
)[:, -1]

actual = df[target].iloc[-len(predicted_prices):].values

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title("BTC Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price (USDT)")
plt.legend()
plt.grid()
plt.show()