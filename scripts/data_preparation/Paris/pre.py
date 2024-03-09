import pandas as pd
import numpy as np
import tqdm
import multiprocessing


def process(num):
    DATASET_NAME = "Paris"
    DATA_FILE_PATH = "datasets/raw_data/{0}/loop_sensor_train.csv".format(DATASET_NAME)

    df = pd.read_csv(DATA_FILE_PATH)

    intersections = df['iu_ac'].unique()
    print(len(intersections))
    times = np.sort(df['t_1h'].unique()) # [:20]

    begin, end = int(len(times)*num/10), int(len(times)*(num+1)/10)
    t_tmp = times[begin: end]
    df_new = pd.DataFrame(columns=['iu_ac'])
    df_new['iu_ac'] = intersections

    for i in tqdm.trange(len(t_tmp), ncols=60):
        t = t_tmp[i]
        df_tmp = df.loc[df['t_1h'] == t][['iu_ac', 'q']]
        df_tmp.rename(columns={'q': str(t)}, inplace=True)
        df_new = pd.merge(df_new, df_tmp, how='left', left_on='iu_ac', right_on='iu_ac')

    index = chr(ord('a') + num)
    df_new.to_csv(f'datasets/raw_data/Paris/transposed/data_transposed_{index}.csv', index=False)


if __name__=='__main__':
    pool = multiprocessing.Pool(10)
    pool.map(process, range(10))
    pool.close()
    pool.join()