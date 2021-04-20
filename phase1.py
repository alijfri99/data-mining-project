import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("divar_posts_dataset.csv")

# correcting a value in the year column, and then converting the year column to numeric
dataset['year'].loc[(dataset['year'] == '<1366')] = '1366'
dataset['year'] = pd.to_numeric(dataset['year'])

print(dataset.columns)

# printing the name of each column and the unique values of each column
for column in dataset.columns:
    if dataset.columns.get_loc(column) == 0:
        continue

    print("Column name:", column)
    print(dataset[column].unique())

    request = input("Would you like to see every unique value for this column?")
    if request == 'y':
        for unique_value in dataset[column].unique():
            print(unique_value)

    is_numeric = input("Is this data numeric?")
    if is_numeric == 'y':
        print("Min:", dataset[column].min())
        print("Max:", dataset[column].max())
        print("Mean:", dataset[column].mean())
        print("Mode:", dataset[column].mode())
        print("Median:", dataset[column].median())
        plt.figure()
        boxplot = sns.boxplot(data=dataset[column])
        plt.show()
    else:
        print("Mode:", dataset[column].mode())
        print("*****\n*****\n*****")
        continue
    print("*****\n*****\n*****")
