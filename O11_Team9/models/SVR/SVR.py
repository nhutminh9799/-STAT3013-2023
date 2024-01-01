#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np #Cho phép làm việc hiệu quả với ma trận và mảng
import pandas as pd #Pandas rất phù hợp với nhiều loại dữ liệu khác nhau. Dữ liệu dạng bảng với các cột được nhập không đồng nhất, như trong bảng SQL hoặc bảng tính Excel.
import statistics as st #cung cấp các hàm để thống kê toán học của dữ liệu số.
import matplotlib.pyplot as plt #biểu đồ
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[15]:


GRAB=pd.read_csv("GRAB.csv",delimiter=',')


# In[16]:


GRAB


# In[17]:


# Chọn các cột dữ liệu cần sử dụng
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'
X = GRAB[features]
y = GRAB[target]


# In[18]:


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


# Xây dựng mô hình SVR
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)


# In[21]:


# Dự đoán giá trị Close trên tập kiểm tra
y_pred = svr.predict(X_test_scaled)


# In[22]:


# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)


# In[23]:


# Tính toán MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Tính toán MAPE trên tập kiểm tra
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)


# In[24]:


# Dự đoán 30 ngày tiếp theo
X_future = X[-30:]  # Lấy 30 ngày gần nhất từ dữ liệu X
X_future_scaled = scaler.transform(X_future)  # Chuẩn hóa dữ liệu
y_future_pred = svr.predict(X_future_scaled)  # Dự đoán 30 ngày tiếp theo


# In[27]:


y_future_pred


# In[25]:


# Tạo dữ liệu cho biểu đồ
y_pred_all = np.concatenate([y_train, y_test, y_pred, y_future_pred])  # Kết hợp dữ liệu dự đoán trong quá khứ và tương lai
dates = pd.date_range(start=GRAB['Date'].iloc[0], periods=len(y_pred_all), freq='D')


# In[26]:


# Vẽ biểu đồ
plt.plot(dates[:len(y_train)], y_train, label='Train')
plt.plot(dates[len(y_train):len(y_train)+len(y_test)], y_test, label='Test')
plt.plot(dates[len(y_train)+len(y_test):len(y_train)+len(y_test)+len(y_pred)], y_pred, label='Predict')
plt.plot(dates[len(y_train)+len(y_test):len(y_train)+len(y_test)+len(y_future_pred)], y_future_pred, label='Predict 30 days')
plt.title('SVR - Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()

