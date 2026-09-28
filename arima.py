"""Временные ряды: декомпозиция, ARIMA, прогноз."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=365)
trend = np.linspace(100, 200, 365)
seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 30)
noise = np.random.normal(0, 5, 365)
series = pd.Series(trend + seasonal + noise, index=dates)

# === 1. Декомпозиция ===
decomp = seasonal_decompose(series, model="additive", period=30)
decomp.plot()
plt.savefig("decomposition.png")
plt.show()

# === 2. Проверка стационарности (ADF) ===
adf_result = adfuller(series.dropna())
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value:       {adf_result[1]:.4f}")
print("Стационарность:", "есть" if adf_result[1] < 0.05 else "нет")

# === 3. Train/Test ===
train = series[:-30]
test = series[-30:]

# === 4. ARIMA ===
model = ARIMA(train, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=30)

mae = mean_absolute_error(test, forecast)
print(f"\nMAE: {mae:.2f}")
print(f"AIC: {model.aic:.2f}")

# === 5. Визуализация ===
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="green")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.legend()
plt.title("ARIMA прогноз")
plt.savefig("forecast.png")
plt.show()
