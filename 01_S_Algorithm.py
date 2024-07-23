import pandas as pd
data = pd.read_csv("C:/Users/91892/Downloads/training_data.csv")
hypothesis = ['0'] * (len(data.columns) - 1)
for index, row in data.iterrows():
    if row[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'
print("Most specific hypothesis found by FIND-S algorithm:")
print(hypothesis)
