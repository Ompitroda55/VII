import pandas as pd

list_data = [10, 20, 30, 40]
list_series = pd.Series(list_data)
print("Series from list:")
print(list_series)
print()

dict_data = {'A': 100, 'B': 200, 'C': 300, 'D': 400}
dict_series = pd.Series(dict_data)
print("Series from dictionary:")
print(dict_series)
print()

import numpy as np
array_data = np.array([1, 2, 3, 4, 5])
array_series = pd.Series(array_data)
print("Series from array:")
print(array_series)
print()

csv_series = pd.read_csv('C:/Users/ompit/Desktop/VII/PP/Labs/sample.csv', header=None, names=['Value'])
print("Series from CSV file:")
print(csv_series)
print()

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 28, 22],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)
print()

df.to_excel('people.xlsx', index=False)
excel_df = pd.read_excel('people.xlsx')
print("DataFrame from Excel file:")
print(excel_df)
