import pandas as pd

# CSV --> PANDAS DATAFRAME --> DICT
df = pd.read_csv('scores.csv', sep=",")
df.drop(df.columns[0], axis=1, inplace=True) # drop first column
print(df.iloc[0])
dict = df.to_dict(orient='records')
print(dict[0]) # print row 0 as dict