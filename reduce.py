import numpy as np
import pandas as pd

data = np.transpose(np.loadtxt('exp_test.csv'))
df = pd.read_csv('exp_test.csv')

print(df.head())

new_data = df.iloc[::10,:]

new_data.to_csv('test_data.csv',index = False)
