import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ========================================================
# Klasa DataLoader – odpowiedzialna za pobieranie i przygotowanie danych
# ========================================================
class DataLoader:
    def __init__(self, symbol, start_date, end_date, window_size=60):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size

    def load_data(self):
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if 'Close' not in data.columns:
            raise ValueError("Pobrane dane nie zawierają kolumny 'Close'.")
        df = data[['Close']]
        df.dropna(inplace=True)
        self.df = df
        return df

    def scale_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.df)
        return self.scaled_data, self.scaler

    def create_sequences(self):
        X, y = [], []
        data = self.scaled_data
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        self.X, self.y = X, y
        # Daty odpowiadają kolejnym etykietom (pomijamy pierwsze window_size obserwacji)
        self.dates = self.df.index[self.window_size:]
        return X, y, self.dates

    def train_test_split(self, train_ratio=0.8):
        train_size = int(len(self.X) * train_ratio)
        X_train, X_test = self.X[:train_size], self.X[train_size:]
        y_train, y_test = self.y[:train_size], self.y[train_size:]
        dates_train, dates_test = self.dates[:train_size], self.dates[train_size:]
        # Dostosowujemy kształt danych dla sieci LSTM/GRU: (próbki, kroki czasowe, cechy)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_train, X_test, y_train, y_test, dates_train, dates_test


# ========================================================
# Klasa bazowa StockModel – definiuje interfejs budowania, trenowania i oceny modelu
# ========================================================
class StockModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        raise NotImplementedError("Metoda build_model() musi być zaimplementowana w klasach dziedziczących.")

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test, scaler):
        predicted = self.model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
        mae = mean_absolute_error(real_prices, predicted_prices)
        return rmse, mae, predicted_prices, real_prices


# ========================================================
# Klasa LSTMModel – buduje model oparty na warstwach LSTM
# ========================================================
class LSTMModel(StockModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.compile_model()


# ========================================================
# Klasa GRUModel – alternatywny model wykorzystujący warstwy GRU
# ========================================================
class GRUModel(StockModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(GRU(units=50, return_sequences=True, input_shape=self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(GRU(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.compile_model()


# ========================================================
# Główna funkcja uruchamiająca pipeline
# ========================================================
def main():
    # Ustawienia
    symbol = "INTC"
    start_date = "2010-01-01"
    end_date = "2025-01-31"
    window_size = 60

    # Pobieranie i przygotowanie danych
    data_loader = DataLoader(symbol, start_date, end_date, window_size)
    data_loader.load_data()
    data_loader.scale_data()
    data_loader.create_sequences()
    X_train, X_test, y_train, y_test, dates_train, dates_test = data_loader.train_test_split(train_ratio=0.8)
    input_shape = (X_train.shape[1], 1)

    # Budowa modeli
    # Model 1: LSTM
    lstm_model = LSTMModel(input_shape)
    lstm_model.build_model()
    # Model 2: GRU
    gru_model = GRUModel(input_shape)
    gru_model.build_model()

    # Trenowanie modeli
    print("Trening modelu LSTM...")
    lstm_history = lstm_model.train(X_train, y_train)
    print("Trening modelu GRU...")
    gru_history = gru_model.train(X_train, y_train)

    # Ocena modeli
    lstm_rmse, lstm_mae, lstm_predicted, real_prices = lstm_model.evaluate(X_test, y_test, data_loader.scaler)
    gru_rmse, gru_mae, gru_predicted, _ = gru_model.evaluate(X_test, y_test, data_loader.scaler)

    print(f"\nWyniki modelu LSTM: RMSE = {lstm_rmse:.4f}, MAE = {lstm_mae:.4f}")
    print(f"Wyniki modelu GRU:  RMSE = {gru_rmse:.4f}, MAE = {gru_mae:.4f}")

    # Fine tuning – wybieramy model z mniejszym RMSE (możesz modyfikować kryterium)
    if lstm_rmse < gru_rmse:
        print("\nModel LSTM radzi sobie lepiej.")
        best_model_predicted = lstm_predicted
    else:
        print("\nModel GRU radzi sobie lepiej.")
        best_model_predicted = gru_predicted

    # Wizualizacja wyników najlepszego modelu
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, real_prices, color='blue', label='Rzeczywiste ceny')
    plt.plot(dates_test, best_model_predicted, color='red', label='Prognozowane ceny (najlepszy model)')
    plt.title('Prognoza cen akcji firmy Intel')
    plt.xlabel('Data')
    plt.ylabel('Cena akcji (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
