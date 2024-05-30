import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from lib.load_dataset import load_wp_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler


def normalize_dataset(data, normalizer):
    shape, dim = data.shape, data.shape[-1]
    data = data.reshape([-1, dim])
    if normalizer == 'max01':
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    data = data.reshape(shape)
    data = scaler.transform(data)
    return data, scaler


class WPDataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                 history: int, horizon: int, mode: str):
        self.data_x = data_x
        self.data_y = data_y
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.his = history
        self.hor = horizon
        self.mode = mode
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must selected in [`train`, `val`, `test`]")

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)
        else:
            return len(self.test_idx)

    def __getitem__(self, item):
        if self.mode == 'train':
            idx = self.train_idx[item]
        elif self.mode == 'val':
            idx = self.val_idx[item]
        else:
            idx = self.test_idx[item]
        x = self.data_x[idx: idx + self.his, ...]
        x = torch.from_numpy(x.astype(np.float32))
        y = self.data_y[idx + self.his: idx + self.his + self.hor, ...]
        y = torch.from_numpy(y.astype(np.float32))
        return x, y


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def get_dataloader(dataset, history, horizon, batch_size,
                       val_ratio=0.2, test_ratio=0.2, normalizer='std'):
    # load raw st dataset
    data_x, data_y, times = load_wp_dataset(dataset)  # (size, node_num, dim)

    # normalize data
    data_x, scaler = normalize_dataset(data_x, normalizer)

    # concat with time
    month_of_year = times[..., 0] - 1  # 0 ~ 11
    day_of_year = times[..., 1] - 1  # 0 ~ 365
    time_of_day = (times[..., 2] * 3600 + times[..., 3] * 60 + times[..., 4]) // 600  # 0 ~ 143
    data_x = np.concatenate([
            data_x,
            time_of_day.reshape([*time_of_day.shape, 1]),
            day_of_year.reshape([*day_of_year.shape, 1]),
            month_of_year.reshape([*month_of_year.shape, 1])
        ],
        axis=-1
    )

    # split dataset by ratio
    random_idx = np.random.permutation(data_x.shape[0] - (history + horizon) + 1)
    train_idx, val_idx, test_idx = split_data_by_ratio(random_idx, val_ratio, test_ratio)

    # get dataset
    train_dataset = WPDataset(data_x, data_y, train_idx, val_idx, test_idx, history, horizon, "train")
    val_dataset = WPDataset(data_x, data_y, train_idx, val_idx, test_idx, history, horizon, "val")
    test_dataset = WPDataset(data_x, data_y, train_idx, val_idx, test_idx, history, horizon, "test")

    # get dataloader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default='WPData', type=str)
    parser.add_argument('--column_wise', default=False, type=bool)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    # train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer='std')
    #
    # length = len(train_dataloader)
    tmp = 1
