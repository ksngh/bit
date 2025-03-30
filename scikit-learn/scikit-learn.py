# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# (1) 시뮬레이션용 데이터 생성
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=500, freq='H')

data = pd.DataFrame({
    'datetime': dates,
    'price': np.cumsum(np.random.randn(500)) + 20000,
    'rsi': np.random.uniform(30, 70, 500),
    'macd': np.random.randn(500),
    'volume_change': np.random.randn(500),
    'obv': np.cumsum(np.random.randn(500)),
    'funding_rate': np.random.uniform(-0.01, 0.01, 500),
    'oi_change': np.random.randn(500),
    'nvt_ratio': np.random.uniform(50, 100, 500),
    'mvrv_zscore': np.random.randn(500),
    'sentiment_score': np.random.uniform(-1, 1, 500)
})

features = ['rsi', 'macd', 'volume_change', 'obv', 'funding_rate', 'oi_change', 'nvt_ratio', 'mvrv_zscore', 'sentiment_score']
target = 'price'

# (2) 데이터 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[features + [target]])

# (3) 시계열 데이터 구조로 변환
sequence_length = 24
X, y = [], []

for i in range(sequence_length, len(scaled)):
    X.append(scaled[i-sequence_length:i, :-1])  # 피처만
    y.append(scaled[i, -1])  # 가격 (target)

X = np.array(X)
y = np.array(y)

# (4) LSTM 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# (5) 모델 학습
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# (6) 예측 수행
predicted = model.predict(X)

# (7) 가격 역정규화
# target만 따로 scaler 다시 정의
price_scaler = MinMaxScaler()
price_scaler.fit(data[[target]])
predicted_prices = price_scaler.inverse_transform(predicted)

# (8) 실제 가격 비교 시각화
actual = data[target].iloc[-len(predicted_prices):].values

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title("Bitcoin Price Prediction (LSTM)")
plt.xlabel("Time (Hourly)")
plt.ylabel("Price (USDT)")
plt.legend()
plt.grid()
plt.show()
