import pandas as pd
import numpy as np
df = pd.read_csv('inputs/table.tex',
                 sep='&',
                 header=None,
                 engine='python')
print(df)
sum_ = []

for index, row in df.iterrows():
    sum_.append(sum(row[1:5]))

print(sum_)
