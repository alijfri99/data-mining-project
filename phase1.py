import pandas as pd

dataset = pd.read_csv("divar_posts_dataset.csv")
pd.set_option('display.max_columns', None)
print(dataset['archive_by_user'].unique())