import pandas as pd
data = pd.read_csv("C:/Users/91892/Downloads/02_training_data.csv")

attributes = list(data.columns)[:-1]
examples = data.values
S = ['0'] * len(attributes)  
G = [['?'] * len(attributes)] 

for example in examples:
    if example[-1] == 'Yes':
        for i in range(len(S)):
            if S[i] == '0':
                S[i] = example[i]
            elif S[i] != example[i]:
                S[i] = '?'
        G = [g for g in G if all(g[i] == '?' or g[i] == example[i] for i in range(len(g)))]
    else:  
        new_G = []
        for g in G:
            for i in range(len(g)):
                if g[i] == '?':
                    new_h = g.copy()
                    new_h[i] = example[i]
                    if all(new_h[j] == '?' or new_h[j] == S[j] for j in range(len(new_h))):
                        new_G.append(new_h)
        G = new_G
print("S: ", S)
print("G: ", G)
