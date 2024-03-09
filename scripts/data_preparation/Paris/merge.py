import pandas as pd
import os


dir = './datasets/raw_data/Paris/transposed/'
files = os.listdir(dir)
print(files)
df = pd.read_csv(dir + files[0])
for f in files[1:]:
    df_tmp = pd.read_csv(dir + f)
    df = pd.merge(df, df_tmp, how='left', left_on='iu_ac', right_on='iu_ac')

df.to_csv('./datasets/raw_data/Paris/transposed.csv', index=False)