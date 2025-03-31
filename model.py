import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def preprocess_data(df, features, target, seq_len=24, predict_gap=1):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features + [target]])

    X, y = [], []
    for i in range(seq_len, len(scaled) - predict_gap):
        X.append(scaled[i-seq_len:i, :-1])         # 피처만 입력
        y.append(scaled[i + predict_gap, -1])      # 미래 시점의 가격을 타겟으로

    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model