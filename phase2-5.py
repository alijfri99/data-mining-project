import pandas as pd
from apyori import apriori

# question 5

dataset = pd.read_csv("divar_posts_dataset.csv")

print(dataset['platform'].isna().sum())
print(dataset['cat1'].isna().sum())
print(dataset['cat2'].isna().sum())
print(dataset['cat3'].isna().sum())

dataset_q1_1 = dataset[['platform', 'cat1']]
print(dataset_q1_1.head())
print(dataset_q1_1.isna().sum())
dataset_q1_1 = dataset_q1_1.values.tolist()
rules_1_1 = apriori(dataset_q1_1, min_support=0.01, min_confidence=0.2)

rules_1_1 = list(rules_1_1)
for i in rules_1_1:
    print(str(rules_1_1.index(i)) + ".", i)

print("\n***********************************************************************************")
print("***********************************************************************************")
print("***********************************************************************************\n")

dataset_q1_2 = dataset[['platform', 'cat2']]
dataset_q1_2 = dataset_q1_2.dropna(subset=['cat2'])
print(dataset_q1_2.head())
print(dataset_q1_2.isna().sum())
dataset_q1_2 = dataset_q1_2.values.tolist()
rules_1_2 = apriori(dataset_q1_2, min_support=0.01, min_confidence=0.2)

rules_1_2 = list(rules_1_2)
for i in rules_1_2:
    print(str(rules_1_2.index(i)) + ".", i)

print("\n***********************************************************************************")
print("***********************************************************************************")
print("***********************************************************************************\n")

dataset_q1_3 = dataset[['platform', 'cat3']]
dataset_q1_3 = dataset_q1_3.dropna(subset=['cat3'])
print(dataset_q1_3.head())
print(dataset_q1_3.isna().sum())
dataset_q1_3 = dataset_q1_3.values.tolist()
rules_1_3 = apriori(dataset_q1_3, min_support=0.01, min_confidence=0.2)

rules_1_3 = list(rules_1_3)
for i in rules_1_3:
    print(str(rules_1_3.index(i)) + ".", i)
