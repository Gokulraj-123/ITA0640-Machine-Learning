import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample sales data with more periods to cover at least two full seasonal cycles
data = {
    'Month': pd.date_range(start='1/1/2020', periods=36, freq='ME'),
    'Sales': [
        120, 135, 148, 160, 155, 165, 170, 175, 180, 190, 195, 200, 
        210, 225, 238, 250, 245, 255, 260, 265, 270, 280, 285, 290, 
        295, 310, 320, 330, 335, 340, 345, 350, 360, 370, 380, 390
    ]
}

df = pd.DataFrame(data)
df.set_index('Month', inplace=True)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df, label='Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Prepare the data for linear regression
X_train = np.array(range(train_size)).reshape(-1, 1)
y_train = train['Sales']
X_test = np.array(range(train_size, len(df))).reshape(-1, 1)
y_test = test['Sales']

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions using linear regression
y_pred_lr = lr_model.predict(X_test)

# Train the Exponential Smoothing model
hw_model = ExponentialSmoothing(train['Sales'], seasonal='add', seasonal_periods=12).fit()

# Make predictions using Exponential Smoothing
y_pred_hw = hw_model.forecast(len(test))

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')

# Evaluate Exponential Smoothing
mse_hw = mean_squared_error(y_test, y_pred_hw)
print(f'Exponential Smoothing MSE: {mse_hw}')

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(test.index, y_pred_lr, label='Linear Regression Prediction')
plt.plot(test.index, y_pred_hw, label='Exponential Smoothing Prediction')
plt.title('Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
