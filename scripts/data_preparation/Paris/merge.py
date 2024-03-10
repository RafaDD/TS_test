import pandas as pd
import os


dir = './datasets/raw_data/Paris/transposed/'
files = os.listdir(dir)
print(files)
df = pd.read_csv(dir + files[0])
for f in files[1:]:
    df_tmp = pd.read_csv(dir + f)
    df = pd.merge(df, df_tmp, how='left', left_on='iu_ac', right_on='iu_ac')

df = df.sort_values(by=['iu_ac'])

# df.to_csv('./datasets/raw_data/Paris/transposed.csv', index=False)


DATASET_NAME = "Paris"
print(len(df['iu_ac']))

DATA_FILE_PATH = "datasets/raw_data/Paris/loop_sensor_test_x.csv"
df_2 = pd.read_csv(DATA_FILE_PATH)

index = df_2['iu_ac'].unique()
print(len(index))
df = df[df['iu_ac'].isin(index)]
print(len(df['iu_ac']))

df.to_csv("./datasets/raw_data/Paris/transposed_shortened.csv", index=False)