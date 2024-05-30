import os
import numpy as np
import pandas as pd
import time


def load_SDWPFDataset(data_path="./data/SDWPF/wtbdata_245days.csv", fill_nan=False):
    if os.path.exists("./data/SDWPF/wtbdata_245days.npz"):
        dict_data = np.load("./data/SDWPF/wtbdata_245days.npz")
        return dict_data['data'], dict_data['data_y'], dict_data['times']

    n_turbine = 134  # number of turbine
    csv_data = pd.read_csv(data_path)

    data = csv_data.iloc[:, 3:]
    data = data.replace(to_replace=np.nan, value=0, inplace=False)
    data.iloc[:, -1] = np.maximum(data.iloc[:, -1], 0)
    data = data.replace(to_replace=0, value=np.nan, inplace=False)
    if fill_nan:
        data = data.fillna(method='ffill', inplace=False)
        data = data.fillna(method='bfill', inplace=False)
    data = data.to_numpy().reshape([n_turbine, -1, 10])  # (134, 35280, 10)
    data = data.transpose([1, 0, 2])  # (35280, 134, 10)

    data_y = data[..., -1]

    # process time
    time_info = csv_data[['Day', 'Tmstamp']]

    def process_str_time(row):
        str_time = "{} {}".format(row['Day'], row['Tmstamp'])
        data_sj = time.strptime(str_time, "%j %H:%M")
        return (
            data_sj.tm_mon,   # 采样点的月份
            data_sj.tm_yday,  # 采样点位于一年中的第几天
            data_sj.tm_hour,  # 时
            data_sj.tm_min,   # 分
            data_sj.tm_sec    # 秒
        )

    times = time_info.apply(process_str_time, axis=1).to_numpy()
    times = np.stack(times).reshape([n_turbine, -1, 5]).transpose([1, 0, 2])
    np.savez("./data/SDWPF/wtbdata_245days", data=data, data_y=data_y, times=times)
    return data, data_y, times


def load_wp_dataset(dataset, fill_nan=True):
    if dataset == 'SDWPF':
        data, data_y, times = load_SDWPFDataset(fill_nan=fill_nan)
    else:
        raise ValueError("Invalid data set: {}".format(dataset))
    return data, data_y, times


if __name__ == '__main__':
   a = np.array([[-1, 2, 3], [-1, -1, 0]])
   a = np.maximum(a, 0)
   tmp = 1
