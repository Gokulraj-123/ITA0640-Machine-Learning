import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return -np.sum((counts/np.sum(counts)) * np.log2(counts/np.sum(counts)))

# Calculate Information Gain
def info_gain(data, feature, target="PlayTennis"):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = sum((counts[i]/np.sum(counts)) * entropy(data[data[feature] == vals[i]][target]) for i in range(len(vals)))
    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, features, target="PlayTennis"):
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    if len(features) == 0:
        return np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
    
    best_feature = max(features, key=lambda feature: info_gain(data, feature))
    tree = {best_feature: {}}
    
    for value in np.unique(data[best_feature]):
        subtree = id3(data[data[best_feature] == value], [f for f in features if f != best_feature], target)
        tree[best_feature][value] = subtree
    
    return tree

# Build the decision tree
tree = id3(df, df.columns[:-1])
print("Decision Tree:", tree)

# Function to classify a new sample
def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    return classify(sample, tree[feature][sample[feature]])

# Sample for classification
sample = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': False}
print("Classification:", classify(sample, tree))
