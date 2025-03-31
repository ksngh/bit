from fetch_data import fetch_bybit_candles
from indicators import add_technical_indicators
from model import preprocess_data, create_lstm_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

features = ['rsi', 'macd', 'volume_change', 'obv', 'funding_rate', 'oi_change', 'nvt_ratio', 'mvrv_zscore', 'sentiment_score']
target = 'close'
predict_gap = 1  # 미래 1시간 뒤 가격 예측

# (1) 데이터 수집 및 전처리
df = fetch_bybit_candles()
df = add_technical_indicators(df)
X, y, scaler = preprocess_data(df, features, target, seq_len=24, predict_gap=predict_gap)

# (2) 모델 구성 및 학습
model = create_lstm_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# (3) 예측
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(predicted), len(features))), predicted))
)[:, -1]

# (4) 실제값 (예측 시점에 맞게 미래 가격으로 조정)
actual = df[target].iloc[24 + predict_gap: 24 + predict_gap + len(predicted_prices)].values
future_timestamps = df['timestamp'].iloc[24 + predict_gap: 24 + predict_gap + len(predicted_prices)].values

# (5) 시각화: 실제 vs 예측 가격 + 예측 곡선 강조
plt.figure(figsize=(12, 6))
plt.plot(future_timestamps, actual, label='Actual Future Price', color='dodgerblue')
plt.plot(future_timestamps, predicted_prices, label='Predicted Future Price', color='orange')
plt.title("BTC Price Prediction (LSTM, +1h Ahead)")
plt.xlabel("Datetime")
plt.ylabel("Price (USDT)")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# (6) 예측 곡선만 따로 시각화
plt.figure(figsize=(12, 4))
plt.plot(future_timestamps, predicted_prices, color='orange')
plt.title("Predicted Future BTC Prices (+1h Ahead)")
plt.xlabel("Datetime")
plt.ylabel("Predicted Price (USDT)")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# (7) 성능 지표 평가
mae = mean_absolute_error(actual, predicted_prices)
mse = mean_squared_error(actual, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted_prices)

print(f"\n📊 예측 정확도 평가:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")