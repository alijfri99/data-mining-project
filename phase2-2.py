import pandas as pd
from apyori import apriori

# question 2

dataset = pd.read_csv("orders.csv")

# preprocessing
print(dataset['ID_Order'].is_unique)
print(dataset.isna().sum())
dataset = dataset.drop(['ID_Customer', 'Quantity_item'], axis=1)
dataset['DateTime_CartFinalize'] = dataset['DateTime_CartFinalize'].str[0:10]
dataset['ID_Order'] = 'Ord' + dataset['ID_Order'].map(str)
dataset['ID_Item'] = 'It' + dataset['ID_Item'].map(str)

# now let's construct the transaction dictionary for each Order ID

transaction_dict = dict()

for index, row in dataset.iterrows():
    if index % 10000 == 0:
        print(index)
    if row['ID_Order'] not in transaction_dict:
        transaction_dict[row['ID_Order']] = {row['ID_Item']}
    else:
        transaction_dict[row['ID_Order']] = transaction_dict[row['ID_Order']].union({row['ID_Item']})

# now let's convert the dictionary to a list of lists

transactions = list()

for i in transaction_dict:
    transactions.append(list(transaction_dict[i]))

rules_2_1 = apriori(transactions, min_support=0.0001, min_confidence=0.0001, min_lift=1.000001)

rules_2_1 = list(rules_2_1)

for i in rules_2_1:
    print(str(rules_2_1.index(i)) + ".", i)

print("\n***********************************************************************************")
print("***********************************************************************************")
print("***********************************************************************************\n")

dataset_q2_2 = dataset[['city_name_fa', 'ID_Item']]
print(dataset_q2_2.head())
print(dataset_q2_2.isna().sum())
dataset_q2_2 = dataset_q2_2.values.tolist()
rules_2_2 = apriori(dataset_q2_2, min_support=0.0001, min_confidence=0.2, min_lift=1.000001)

rules_2_2 = list(rules_2_2)
for i in rules_2_2:
    print(str(rules_2_2.index(i)) + ".", i)

print("\n***********************************************************************************")
print("***********************************************************************************")
print("***********************************************************************************\n")

dataset_q2_3 = dataset[['DateTime_CartFinalize', 'ID_Item']]
print(dataset_q2_3.head())
print(dataset_q2_3.isna().sum())
dataset_q2_3 = dataset_q2_3.values.tolist()
rules_2_3 = apriori(dataset_q2_3, min_support=0.0001, min_confidence=0.1, min_lift=1.000001)

rules_2_3 = list(rules_2_3)
for i in rules_2_3:
    print(str(rules_2_3.index(i)) + ".", i)
