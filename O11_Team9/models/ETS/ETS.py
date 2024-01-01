#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


# In[40]:


# Đọc dữ liệu từ file CSV
GRAB = pd.read_csv("GRAB.csv", delimiter=',')
GRAB['Date'] = pd.to_datetime(GRAB['Date'])
GRAB.set_index('Date', inplace=True)


# In[41]:


GRAB


# In[42]:


# Chuẩn bị dữ liệu cho Mann-Kendall Test và Sen's Slope Test
dependent_variable = GRAB['Close']
n = len(dependent_variable)
s = 0
for i in range(n - 1):
    for j in range(i + 1, n):
        s += np.sign(dependent_variable[j] - dependent_variable[i])
mk_statistic = s / (n * (n - 1) / 2)
time = GRAB.index.map(pd.Timestamp.toordinal)
ss_result = linregress(time, dependent_variable)


# In[43]:


print('Mann-Kendall Test:')
print('Statistic:', mk_statistic)
print("Sen's Slope Test:")
print('Slope:', ss_result.slope)
print('Intercept:', ss_result.intercept)
print('p-value:', ss_result.pvalue)


# In[44]:


# Tính ACF
close_price = GRAB['Close']
acf_result = acf(close_price)

print("Autocorrelation Function (ACF):")
print(acf_result)


# In[45]:


# Tính ADF Test
adf_result = adfuller(close_price)

print("Augmented Dickey-Fuller (ADF) Test:")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(key, ":", value)


# In[46]:


# Phân tích mùa vụ
time_series = GRAB['Close']
decomposition = seasonal_decompose(time_series, model='additive', period=12)
decomposition.plot()
plt.show()


# In[47]:


# Xây dựng mô hình ETS
seasonal_periods = 12
model = ETSModel(time_series, error="add", trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
print(model.summary())


# In[48]:


# Dự báo
n = 30
forecast = model.forecast(steps=n)
forecast_dates = pd.date_range(start=close_price.index[-1], periods=n+1, closed='right')[1:]


# In[49]:


data = pd.concat([close_price, pd.Series(forecast, index=forecast_dates)], axis=0)
data = pd.DataFrame(data, columns=['Close'])


# In[50]:


# Tính RMSE, MAE và MAPE
actual_values = close_price[-n:].values
forecast = forecast[-n:]
rmse = np.sqrt(mean_squared_error(actual_values, forecast))
mae = mean_absolute_error(actual_values, forecast)
mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)


# In[51]:


# Chuẩn bị dữ liệu cho biểu đồ
start_date = GRAB.index[-1] + pd.DateOffset(days=1)
end_date = start_date + pd.DateOffset(days=n_days-1)
forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')


# In[52]:


# Vẽ biểu đồ
plt.plot(GRAB.index, GRAB['Close'], label='Train + Test')
plt.plot(forecast_dates, forecast, label='Forecast')
plt.plot(forecast_dates, actual_values, label='Actual')
plt.title('ETS - Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[ ]:




