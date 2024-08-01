import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a simple synthetic car dataset
data = {
    'buying': ['v-high', 'high', 'med', 'low', 'v-high', 'high', 'med', 'low'],
    'maint': ['v-high', 'high', 'med', 'low', 'v-high', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5-more', '2', '3', '4', '5-more'],
    'persons': ['2', '4', 'more', '2', '4', 'more', '2', '4'],
    'lug_boot': ['small', 'med', 'big', 'small', 'med', 'big', 'small', 'med'],
    'safety': ['low', 'med', 'high', 'low', 'med', 'high', 'low', 'med'],
    'class': ['unacc', 'acc', 'good', 'v-good', 'unacc', 'acc', 'good', 'v-good']
}

car_df = pd.DataFrame(data)

# One-hot encoding categorical variables
car_df_encoded = pd.get_dummies(car_df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

X = car_df_encoded.drop('class', axis=1)
y = car_df['class']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Building and training the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred, labels=['unacc', 'acc', 'good', 'v-good'])

print("Confusion Matrix:\n", confusion)
print("Accuracy:", accuracy)

# Plotting the confusion matrix
plt.figure(figsize=(8, 5))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['unacc', 'acc', 'good', 'v-good'], yticklabels=['unacc', 'acc', 'good', 'v-good'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
