import pandas as pd

dataset = pd.read_csv("divar_posts_dataset.csv")

# method 1
print("Method 1:")
completed_dataset = dataset.dropna(axis='index')
print(len(completed_dataset.index))
print(len(dataset.index))
print("Result =", len(completed_dataset.index)/len(dataset.index))
print("*****")

# method 2
print("Method 2:")
dataset_m2 = dataset[['brand', 'cat1', 'cat2', 'cat3', 'city', 'created_at', 'desc', 'platform', 'price', 'title']]
completed_dataset_m2 = dataset_m2.dropna(axis='index')
print(len(completed_dataset_m2.index))
print(len(dataset_m2.index))
print("Result =", len(completed_dataset_m2.index)/len(dataset_m2.index))
print("*****")

# method 3
print("Method 3:")
total_data = len(dataset.columns) * len(dataset.index)
number_of_actual_values = total_data - dataset.isna().sum().sum()
print(number_of_actual_values)
print(total_data)
print("Result =", number_of_actual_values/total_data)
