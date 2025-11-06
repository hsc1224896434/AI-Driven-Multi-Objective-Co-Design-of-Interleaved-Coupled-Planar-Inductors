from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
import json
import os
import numpy as np


class EmptyDataset(Dataset):
    def __init__(self):
        super(EmptyDataset, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset, no items to get")


class MagNetDataModule(pl.LightningDataModule):
    def __init__(self, data_B, data_F, data_T, data_H_dc, data_P, norm_info_path=None,
                 batch_size=1, num_workers=1, sample_num=1024):
        super().__init__()
        # data are loaded using pd.read_csv(in_file, header=None)
        self.data_B = data_B
        self.data_F = data_F
        self.data_T = data_T
        self.data_H_dc = data_H_dc
        self.data_P = data_P
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_num = sample_num  # sample number for each data point.

        if not norm_info_path is None:
            if os.path.exists(norm_info_path):
                with open(norm_info_path, 'r') as file:
                    self.norm_info = json.load(file)
            else:
                # 如果标准化参数文件不存在，先进行数据处理并保存参数
                self.prepare_data()
                norm_info = {
                    'normB': [self.normB[0].item(), self.normB[1].item()],
                    'normF': [self.normF[0].item(), self.normF[1].item()],
                    'normT': [self.normT[0].item(), self.normT[1].item()],
                    'normH_dc': [self.normH_dc[0].item(), self.normT[1].item()],
                    'normP': [self.normP[0].item(), self.normP[1].item()]
                }
                with open(norm_info_path, 'w') as f:
                    json.dump(norm_info, f)
                self.norm_info = norm_info
        else:
            self.norm_info = None

    def prepare_data(self):
        self.data_B = self.data_B.astype(np.float32)
        in_B = torch.from_numpy(self.data_B.values).float().unsqueeze(2)  # 添加维度，2为第三个维度
        # downsample
        N, D, C = in_B.size()
        in_B = torch.nn.functional.interpolate(in_B.view(N, C, D), size=self.sample_num, mode='linear').view(N, -1,
                                                                                                             C)  # 线性插值

        in_T = torch.from_numpy(self.data_T.values).float().view(-1, 1)

        in_H_dc = torch.from_numpy(self.data_H_dc.values).float().view(-1, 1)

        in_F = self.data_F
        in_F = torch.from_numpy(in_F.values).float().view(-1, 1)

        # Transform
        in_F = torch.log(in_F)

        if self.norm_info == None:
            self.normB = [torch.mean(in_B), torch.std(in_B)]
            self.normF = [torch.mean(in_F), torch.std(in_F)]
            self.normT = [torch.mean(in_T), torch.std(in_T)]
            self.normH_dc = [torch.mean(in_H_dc), torch.std(in_H_dc)]
        else:
            self.normB = self.norm_info['normB']
            self.normF = self.norm_info['normF']
            self.normT = self.norm_info['normT']
            self.normH_dc = self.norm_info['normH_dc']

        # Normalize
        in_B = (in_B - self.normB[0]) / self.normB[1]
        in_F = (in_F - self.normF[0]) / self.normF[1]
        in_T = (in_T - self.normT[0]) / self.normT[1]
        in_H_dc = (in_H_dc - self.normH_dc[0]) / self.normH_dc[1]

        if not self.data_P is None:
            gt_P = torch.from_numpy(self.data_P.values).float().view(-1, 1)
            out_P = torch.log(gt_P)
            self.normP = [torch.mean(out_P), torch.std(out_P)]
            out_P = (out_P - self.normP[0]) / self.normP[1]
        else:
            # fake ground truth
            gt_P = torch.zeros((N, 1))
            out_P = torch.zeros((N, 1))
            assert self.norm_info != None, 'norm_info is nessesary when groundtruth is not provided!'
            self.normP = self.norm_info['normP']

        print('Log for sample dims:')
        print(in_B.size())  # torch.Size([40712, 1024, 1])
        print(in_T.size())  # torch.Size([40712, 1])
        print(in_F.size())  # torch.Size([40712, 1])
        print(in_H_dc.size())  # torch.Size([40712, 1])
        print(out_P.size())  # torch.Size([40712, 1])
        print(gt_P.size())  # torch.Size([40712, 1])

        self.dataset = TensorDataset(in_B, in_F, in_T, in_H_dc, out_P, gt_P)

        # 保存标准化参数为JSON文件
        norm_info = {
            'normB': [self.normB[0].item(), self.normB[1].item()],
            'normF': [self.normF[0].item(), self.normF[1].item()],
            'normT': [self.normT[0].item(), self.normT[1].item()],
            'normH_dc': [self.normH_dc[0].item(), self.normH_dc[1].item()],
            'normP': [self.normP[0].item(), self.normP[1].item()]
        }
        norm_info_path = "norm_info.json"  # 这里可以根据实际需求指定保存路径和文件名
        with open(norm_info_path, 'w') as f:
            json.dump(norm_info, f)

    def setup(self, stage, train_ratio=0.8, val_ratio=0.2):
        if stage == 'fit':
            train_size = int(train_ratio * len(self.dataset))
            valid_size = int(val_ratio * len(self.dataset))
            test_size = len(self.dataset) - train_size - valid_size
            (
                self.train_dataset,
                self.valid_dataset,
                self.test_dataset,
            ) = torch.utils.data.random_split(
                self.dataset, [train_size, valid_size, test_size]
            )
        elif stage == 'inference':
            train_size, valid_size, test_size = 0, 0, len(self.dataset)

            self.train_dataset = EmptyDataset()
            self.valid_dataset = EmptyDataset()
            self.test_dataset = self.dataset

        print(rf'Split the dataset: Train({train_size}) | Val({valid_size}) | Test({test_size})')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)