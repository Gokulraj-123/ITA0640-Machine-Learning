import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('E:/Machine learning/houseprice.csv')

# Define features and target
X = data[['bedrooms', 'square_feet', 'year_built']]
y = data['price']

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = ((y_test - y_pred) ** 2).mean()
print(f"Mean Squared Error: {mse}")

# Show predictions for test set
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(results)
