import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("divar_posts_dataset.csv")
dataset.boxplot()

# printing the name of each column and the unique values of each column
for column in dataset.columns:
    if dataset.columns.get_loc(column) == 0:
        continue
    print(column)
    print(dataset[column].unique())
    is_nominal = input("Is this data numeric?")
    if is_nominal == 'y':
        print("Min:", dataset[column].min())
        print("Max:", dataset[column].max())
        print("Mean:", dataset[column].mean())
        print("Mode:", dataset[column].mode())
        print("Median:", dataset[column].median())
    else:
        print("Mode:", dataset[column].mode())
        print("*****\n*****\n*****")
        continue
    print("*****\n*****\n*****")
