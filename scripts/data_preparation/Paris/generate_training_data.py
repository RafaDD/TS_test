import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    # graph_file_path = args.graph_file_path
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.

    # read data
    df = pd.read_csv(data_file_path)
    mean = df.mean(axis=1)
    for i, col in enumerate(df):
        df.iloc[:, i] = df.iloc[:, i].fillna(mean)
    # print(df[42:50])
    # df.fillna(0, inplace=True)

    intersections = df['iu_ac'] # intersection info

    df.drop('iu_ac', axis=1, inplace=True)

    data = np.array(df).T
    data = np.expand_dims(data, axis=-1)

    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))
    
    cols = np.array(df.columns)
    # print(cols[:10])

    # split data
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = 8484 - 30 * 24
    valid_num = 30 * 24
    test_num = num_samples - train_num - valid_num
    test_gap = 29

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    # test_index = index_list[train_num: train_num + test_num: test_gap]
    valid_index = index_list[train_num:train_num+valid_num]
    
    def add_zero(t):
            if t < 10:
                t = '0' + str(t)
            return str(t)

    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    def get_past_time(t):
        h = int(t[11: 13]) - 1
        d = int(t[8: 10])
        m = int(t[5: 7])
        y = int(t[:4])
        if h < 0:
            h = 23
            d -= 1
            if d == 0:
                m -= 1
                if m == 0:
                    m = 12
                    y -= 1
                d = days[m - 1]
        return str(y)+'-'+add_zero(m)+'-'+add_zero(d)+' '+add_zero(h)+':00:00'        

    print(cols[8484])
    test_ids = {}
    test_index = []
    test_df = pd.read_csv('datasets/Paris/loop_sensor_test_x.csv')
    times = np.sort(test_df['t_1h'].unique())
    i = 0
    for t in times:
        ids = np.array(test_df.loc[test_df["t_1h"] == t]['iu_ac'])
        for id in ids:
            if id in test_ids.keys():
                test_ids[id].append(i)
            else:
                test_ids[id] = [i]
        # print(1, test_ids[-1])
        t_past = get_past_time(t)
        index = int(np.where(cols == t_past)[0][0] + 1)
        test_index.append((index - history_seq_len, index, index + future_seq_len))
        i += 1
        # print(2, test_index[-1])

    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(len(test_index)))

    print(1, train_index[:10])
    print(2, test_index[:10])
    
    np.save('datasets/Paris/test_ids.npy', test_ids)

    # normalize data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)

    # =============================
    # Graph structure hasn't been processed
    # =============================
    # shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 24
    FUTURE_SEQ_LEN = 1

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]                   # target channel(s)

    DATASET_NAME = "Paris"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True                  # if add day_of_month feature
    DOY = True                  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/transposed_shortened.csv".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
