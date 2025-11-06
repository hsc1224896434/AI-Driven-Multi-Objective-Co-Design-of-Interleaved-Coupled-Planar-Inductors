# -*- encoding: utf-8 -*-
'''
Filename         :Model_Validation.py
Description      :Validate the model on validation set (10%) from Training data.
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
DATA_ROOT = r'F:\Coreloss\Database 15 Material\Single Cycle\77'  # The data directory in the test PC

def load_dataset(in_file1=DATA_ROOT + rf'\B.csv',
                 in_file2=DATA_ROOT + rf'\F.csv',
                 in_file3=DATA_ROOT + rf'\T.csv',
                 in_file4=DATA_ROOT + rf'\Hdc.csv',
                 in_file5=DATA_ROOT + rf'\P.csv'):
    # Load data with proper handling of headers and missing values
    data_B = pd.read_csv(in_file1, header=None)
    data_F = pd.read_csv(in_file2, header=None)
    data_T = pd.read_csv(in_file3, header=None)
    data_H_dc = pd.read_csv(in_file4, header=None, skiprows=2)  # Skip 2 rows if necessary
    data_P = pd.read_csv(in_file5, header=None)

    # Ensure all datasets have the same number of samples
    min_samples = min(len(data_B), len(data_F), len(data_T), len(data_H_dc), len(data_P))
    data_B = data_B.iloc[:min_samples]
    data_F = data_F.iloc[:min_samples]
    data_T = data_T.iloc[:min_samples]
    data_H_dc = data_H_dc.iloc[:min_samples]
    data_P = data_P.iloc[:min_samples]

    return data_B, data_F, data_T, data_H_dc, data_P

#%%

def core_loss(data_B, data_F, data_T, data_H_dc, data_P, SAVE_RESULT):

    # Create Pytorch Lightning Dataset
    print('------------/ Start load dataset... /------------')
    dm = MagNetDataModule(data_B, data_F, data_T, data_H_dc, data_P, batch_size=128,
                          norm_info_path=None)
    dm.prepare_data()
    dm.setup('fit', train_ratio=0.9, val_ratio=0.1)
    print('------------/ Successfully load dataset! /------------')

    # Prepare Model
    print('------------/ Start prepare model...  /------------')
    net = Transformer()
    model = Lit_model(net, normP=dm.normP, save_dir=rf'./Result/')
    torchinfo.summary(model)  # print Mem. for each layer

    trainer = pl.Trainer(
        accelerator="cpu",
        benchmark=False,
        deterministic=True,
        # precision='16-mixed',
        logger=False
    )
    print('------------/ Successfully load model! /------------')

    # Inference
    print('------------/ Start inference... /------------')
    trainer.test(model, dataloaders=dm.val_dataloader(), ckpt_path=rf'./Model/{Material}_model.ckpt')

    # Save Results
    if SAVE_RESULT:
        os.makedirs('./Result', exist_ok=True)
        data_P = model.results
        with open(rf'./Result/Volumetric_Loss_{Material}_Validation.csv', "w") as f:
            np.savetxt(f, data_P)
            f.close()

    print('------------/ Model validation is finished! /------------')

#%%

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pl.seed_everything(666)  # reproducibility
    SAVE_RESULT = True

    data_B, data_F, data_T, data_H_dc, data_P = load_dataset()

    # Check dataset dimensions before passing to core_loss
    print("Dataset dimensions:")
    print(f"data_B: {data_B.shape}")
    print(f"data_F: {data_F.shape}")
    print(f"data_T: {data_T.shape}")
    print(f"data_H_dc: {data_H_dc.shape}")
    print(f"data_P: {data_P.shape}")

    core_loss(data_B, data_F, data_T, data_H_dc, data_P, SAVE_RESULT)

if __name__ == '__main__':
    main()