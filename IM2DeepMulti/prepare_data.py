import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import wandb
import os
import pandas as pd
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import copy

def prepare_data(config):
    # Get data
    ccs_df_train = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_train_OnlyMoreMultimodals.pkl', 'rb'))
    Train_AtomEnc = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_train_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Train_Globals = pickle.load(
        open("/home/robbe/IM2DeepMulti/data/X_train_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb")
    )
    Train_DiAminoAtomEnc = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_train_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    Train_OneHot = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_train_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_train = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_train-OnlyMoreMultimodals.pickle', 'rb'))

    y_train = np.vstack(y_train)


    # Valid
    ccs_df_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_valid_OnlyMoreMultimodals.pkl', 'rb'))
    Valid_AtomEnc = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_valid_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Valid_Globals = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_valid_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb"))
    Valid_DiAminoAtomEnc = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_valid_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    Valid_OneHot = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_valid_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_valid-OnlyMoreMultimodals.pickle', 'rb'))

    y_valid = np.vstack(y_valid)

    # Test
    ccs_df_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_test_OnlyMoreMultimodals.pkl', 'rb'))
    Test_AtomEnc = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_test_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Test_Globals = pickle.load(
        open("/home/robbe/IM2DeepMulti/data/X_test_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb")
    )
    Test_DiAminoAtomEnc = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_test_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    Test_OneHot = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_test_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_test-OnlyMoreMultimodals.pickle', 'rb'))

    y_test = np.vstack(y_test)
    print(y_test.shape)

    # Convert the data to PyTorch tensors
    Train_AtomEnc = torch.tensor(Train_AtomEnc, dtype=torch.float32)
    Train_Globals = torch.tensor(Train_Globals, dtype=torch.float32)
    Train_DiAminoAtomEnc = torch.tensor(Train_DiAminoAtomEnc, dtype=torch.float32)
    Train_OneHot = torch.tensor(Train_OneHot, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    Valid_AtomEnc = torch.tensor(Valid_AtomEnc, dtype=torch.float32)
    Valid_Globals = torch.tensor(Valid_Globals, dtype=torch.float32)
    Valid_DiAminoAtomEnc = torch.tensor(Valid_DiAminoAtomEnc, dtype=torch.float32)
    Valid_OneHot = torch.tensor(Valid_OneHot, dtype=torch.float32)
    y_valid = torch.tensor(y_valid)

    Test_AtomEnc = torch.tensor(Test_AtomEnc, dtype=torch.float32)
    Test_Globals = torch.tensor(Test_Globals, dtype=torch.float32)
    Test_DiAminoAtomEnc = torch.tensor(Test_DiAminoAtomEnc, dtype=torch.float32)
    Test_OneHot = torch.tensor(Test_OneHot, dtype=torch.float32)
    y_test = torch.tensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(Train_AtomEnc, Train_DiAminoAtomEnc, Train_Globals, Train_OneHot, y_train)
    valid_dataset = TensorDataset(Valid_AtomEnc, Valid_DiAminoAtomEnc, Valid_Globals, Valid_OneHot, y_valid)
    test_dataset = TensorDataset(Test_AtomEnc, Test_DiAminoAtomEnc, Test_Globals, Test_OneHot, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=31)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=31)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=31)

    del Train_AtomEnc, Train_Globals, Train_DiAminoAtomEnc, Train_OneHot, y_train
    del Valid_AtomEnc, Valid_Globals, Valid_DiAminoAtomEnc, Valid_OneHot, y_valid
    del Test_AtomEnc, Test_Globals, Test_DiAminoAtomEnc, Test_OneHot, y_test
    del train_dataset, valid_dataset, test_dataset

    return ccs_df_train, train_loader, ccs_df_valid, valid_loader, ccs_df_test, test_loader
