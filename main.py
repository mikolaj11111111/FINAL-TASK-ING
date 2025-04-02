import pandas as pd

df = pd.read_csv('Main_task_data_training.csv', encoding='ISO-8859-1', sep=';', on_bad_lines='skip')

"""
print("HEAD")
print(df.head())
print('==" * 100')
print("SHAPE")
print(df.shape)
print("==" * 100)
print("INFO")
print(df.info())
print("==" * 100)
print(df)
print(df.columns)

"""
#print(len(df.columns))
for column in df.columns:
    print(f"{column}:", df[column].count())