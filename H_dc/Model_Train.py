# -*- encoding: utf-8 -*-
'''
Filename         :Model_Training.py
Description      :Training the model on Training data.
Author           :Fuzhou University
'''

import os
from typing import Any
import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torchinfo

from MagNet_Data import MagNetDataModule
from MagNet_Model import Transformer, Lit_model


Material = '77'
#%% Load Dataset
DATA_ROOT = fr'F:\Coreloss\Database 15 Material\Single Cycle\{Material}'

def load_dataset(in_file1=DATA_ROOT + rf'\B.csv',
                 in_file2=DATA_ROOT + rf'\F.csv',
                 in_file3=DATA_ROOT + rf'\T.csv',
                 in_file4=DATA_ROOT + rf'\Hdc.csv',
                 in_file5=DATA_ROOT + rf'\P.csv'):
    #data_B = pd.read_csv(in_file1, header=None, sep='\t')
    data_B = pd.read_csv(in_file1, header=None)
    data_F = pd.read_csv(in_file2, header=None)
    data_T = pd.read_csv(in_file3, header=None)
    data_H_dc = pd.read_csv(in_file4, header=None)
    data_P = pd.read_csv(in_file5, header=None)


    return data_B, data_F, data_T, data_H_dc, data_P

# DATA_ROOT = r'E:\MagNet\pre-training\pre-training\3F4'  # The data directory for training data
#
# def load_dataset(in_file1=DATA_ROOT + rf'\B_waveform[T].csv',
#                  in_file2=DATA_ROOT + rf'\Frequency[Hz].csv',
#                  in_file3=DATA_ROOT + rf'\Temperature[C].csv',
#                  in_file4=DATA_ROOT + rf'\Volumetric_losses[Wm-3].csv'):
#     data_B = pd.read_csv(in_file1, header=None)
#     data_F = pd.read_csv(in_file2, header=None)
#     data_T = pd.read_csv(in_file3, header=None)
#     data_P = pd.read_csv(in_file4, header=None)
#
#     return data_B, data_F, data_T, data_P
#%%
def core_loss(data_B, data_F, data_T, data_H_dc, data_P):
    # Create Pytorch Lightning Dataset
    print('------------/ Start load dataset... /------------')
    dm = MagNetDataModule(data_B, data_F, data_T, data_H_dc, data_P, batch_size=128,
                          norm_info_path=None)
    dm.prepare_data()
    dm.setup('fit')  # 设置为训练阶段的数据集划分
    print('------------/ Successfully load dataset! /------------')

    # Prepare Model
    print('------------/ Start prepare model...  /------------')
    net = Transformer()
    model = Lit_model(net, learning_rate=0.001, CLR_step_size=10, normF=dm.normF, normP=dm.normP)#0.003
    torchinfo.summary(model)  # print Mem. for each layer

    trainer = pl.Trainer(
        accelerator="cpu",
        benchmark=False,
        deterministic=True,
        #precision='16-mixed',
        logger=True,  # 开启日志记录，便于查看训练过程指标
        max_epochs=3,  # 设置训练的最大轮数，可根据需要调整
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath='./checkpoints',  # 设置保存检查点的目录
            save_top_k=1,  # 保存最佳的3个检查点
            monitor='val_loss',  # 根据验证损失来判断是否保存检查点
            mode='min'  # 因为是监控验证损失，所以希望其最小化
        )]
    )
    print('------------/ Successfully load model! /------------')

    # Training
    print('------------/ Start training... /------------')
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    print('------------/ Training is finished! /------------')

    # Save the best model (you can also choose to save the final trained model in a different way)
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        os.makedirs('./Model', exist_ok=True)
        os.rename(best_model_path, f'./Model/{Material}_model.ckpt')

#%%
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pl.seed_everything(666)  # reproducibility

    data_B, data_F, data_T, data_H_dc, data_P = load_dataset()

    core_loss(data_B, data_F, data_T, data_H_dc, data_P)

if __name__ == '__main__':
    main()