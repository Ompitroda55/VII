import pandas as pd

data_list = [10, 20, 30, 40]
series_from_list = pd.Series(data_list)

data_dict = {'A': 10, 'B': 20, 'C': 30, 'D': 40}
series_from_dict = pd.Series(data_dict)

import numpy as np
data_array = np.array([1, 2, 3, 4])
series_from_array = pd.Series(data_array)

csv_data = pd.read_csv('example.csv')
series_from_csv = csv_data['ColumnName']

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22]}
df_from_dict = pd.DataFrame(data)

df_from_dict['City'] = ['New York', 'San Francisco', 'Los Angeles']

df_from_dict.drop('Age', axis=1, inplace=True)

data_from_csv = pd.read_csv('data.csv')

df_from_dict.to_csv('new_data.csv', index=False)

data_from_excel = pd.read_excel('data.xlsx')

df_from_dict.to_excel('new_data.xlsx', index=False)

print("Series from List:")
print(series_from_list)

print("\nSeries from Dictionary:")
print(series_from_dict)

print("\nSeries from NumPy Array:")
print(series_from_array)

print("\nSeries from CSV:")
print(series_from_csv)

print("\nDataFrame from Dictionary:")
print(df_from_dict)

print("\nData imported from CSV:")
print(data_from_csv)

print("\nData imported from Excel:")
print(data_from_excel)
