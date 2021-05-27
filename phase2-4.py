import pandas as pd
from apyori import apriori

# question 4

dataset = pd.read_csv("divar_posts_dataset.csv")

dataset = dataset[['platform', 'created_at']]
print(dataset.head())
print(dataset.isna().sum())
dataset = dataset.values.tolist()
rules = apriori(dataset, min_support=0.001, min_confidence=0.2, min_lift=1.000001)

rules = list(rules)
for i in rules:
    print(str(rules.index(i)) + ".", i)
