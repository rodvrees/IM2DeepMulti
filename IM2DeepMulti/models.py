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
import lightning as L
import copy
from utils import LRelu_with_saturation, MultiOutputLoss, MeanMAE, LowestMAE, BASEMODELCONFIG, MeanMAESorted, LowestMAESorted, MeanPearsonRSorted, FlexibleLossSorted

class DeepLCNN(nn.Module):
    def __init__(self, config):
        super(DeepLCNN, self).__init__()
        # self.config = config
        self.config = config
        self.ConvAtomComp = nn.ModuleList()
        # AtomComp input size is batch_size x 60 x 6 but should be batch_size x 6 x 60
        self.ConvAtomComp.append(nn.Conv1d(6, config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same')) #Input is probably 256 now?
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same')) #Input is probably 128 now?
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//4, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        # Flatten
        self.ConvAtomComp.append(nn.Flatten())

        ConvAtomCompSize = (60 // (2 * config['AtomComp_MaxPool_kernel_size'])) * (config['AtomComp_out_channels_start']//4)
        print(ConvAtomCompSize)

        self.ConvDiatomComp = nn.ModuleList()
        # DiatomComp input size is batch_size x 30 x 6 but should be batch_size x 6 x 30
        self.ConvDiatomComp.append(nn.Conv1d(6, config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.MaxPool1d(config['DiatomComp_MaxPool_kernel_size'], config['DiatomComp_MaxPool_kernel_size']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same')) #Input is probably 64 now?
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start']//2, config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        # Flatten
        self.ConvDiatomComp.append(nn.Flatten())

        # Calculate the output size of the DiatomComp layers
        ConvDiAtomCompSize = (30 // config['DiatomComp_MaxPool_kernel_size']) * (config['DiatomComp_out_channels_start']//2)
        print(ConvDiAtomCompSize)

        self.ConvGlobal = nn.ModuleList()
        # Global input size is batch_size x 60
        self.ConvGlobal.append(nn.Linear(60, config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))

        # Calculate the output size of the Global layers
        ConvGlobal_output_size = config['Global_units']
        print(ConvGlobal_output_size)

        # One-hot encoding
        self.OneHot = nn.ModuleList()
        self.OneHot.append(nn.Conv1d(20, config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.Conv1d(config['OneHot_out_channels'], config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.MaxPool1d(config['OneHot_MaxPool_kernel_size'], config['OneHot_MaxPool_kernel_size']))
        self.OneHot.append(nn.Flatten())

        # Calculate the output size of the OneHot layers
        conv_output_size_OneHot = ((60 // config['OneHot_MaxPool_kernel_size']) * config['OneHot_out_channels'])
        print(conv_output_size_OneHot)

        # Calculate the total input size for the Concat layer
        total_input_size = ConvAtomCompSize + ConvDiAtomCompSize + ConvGlobal_output_size + conv_output_size_OneHot
        print(total_input_size)

        # Get total input size as a class variable
        self.total_input_size = total_input_size

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        CNNoutput = torch.cat((atom_comp, diatom_comp, global_feats, one_hot), dim=1)
        return CNNoutput

class DeepLConcat(nn.Module):
    def __init__(self, config, total_input_size):
        super(DeepLConcat, self).__init__()
        self.config = config
        self.total_input_size = total_input_size
        self.Concat = nn.ModuleList()
        self.Concat.append(nn.Linear(self.total_input_size, config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], 1))

    def forward(self, CNNoutput):
        for layer in self.Concat:
            CNNoutput = layer(CNNoutput)

        Concatoutput = CNNoutput
        return Concatoutput

class IM2DeepMulti(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepMulti, self).__init__()
        self.config = config
        self.DeepLCNN = DeepLCNN(self.config)
        self.total_input_size = self.DeepLCNN.total_input_size
        self.DeepLConcat1 = DeepLConcat(self.config, self.total_input_size)
        self.DeepLConcat2 = DeepLConcat(self.config, self.total_input_size)
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN(AtomEnc, DiAtomEnc, Globals, OneHot)
        y_hat1 = self.DeepLConcat1(CNN_output)
        y_hat2 = self.DeepLConcat2(CNN_output)

        # Y_hats are of shape (batch_size, 1) but should be (batch_size, )
        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAE(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAE(y1, y2, y_hat1, y_hat2)

        self.log('Train Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN(AtomEnc, DiAtomEnc, Globals, OneHot)
        y_hat1 = self.DeepLConcat1(CNN_output)
        y_hat2 = self.DeepLConcat2(CNN_output)

        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAE(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAE(y1, y2, y_hat1, y_hat2)

        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN(AtomEnc, DiAtomEnc, Globals, OneHot)
        y_hat1 = self.DeepLConcat1(CNN_output)
        y_hat2 = self.DeepLConcat2(CNN_output)

        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAE(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAE(y1, y2, y_hat1, y_hat2)

        self.log('Test Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        CNN_output = self.DeepLCNN(AtomEnc, DiAtomEnc, Globals, OneHot)
        y_hat1 = self.DeepLConcat1(CNN_output)
        y_hat2 = self.DeepLConcat2(CNN_output)

        return torch.hstack([y_hat1, y_hat2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

class IM2Deep(nn.Module):
    def __init__(self, config):
        super(IM2Deep, self).__init__()
        # self.config = config
        self.config = config
        self.ConvAtomComp = nn.ModuleList()
        # AtomComp input size is batch_size x 60 x 6 but should be batch_size x 6 x 60
        self.ConvAtomComp.append(nn.Conv1d(6, config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start'], config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start'], config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same')) #Input is probably 256 now?
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//2, config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.MaxPool1d(config['AtomComp_MaxPool_kernel_size'], config['AtomComp_MaxPool_kernel_size']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//2, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same')) #Input is probably 128 now?
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvAtomComp.append(nn.Conv1d(config['AtomComp_out_channels_start']//4, config['AtomComp_out_channels_start']//4, config['AtomComp_kernel_size'], padding='same'))
        self.ConvAtomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        # Flatten
        self.ConvAtomComp.append(nn.Flatten())

        ConvAtomCompSize = (60 // (2 * config['AtomComp_MaxPool_kernel_size'])) * (config['AtomComp_out_channels_start']//4)
        print(ConvAtomCompSize)

        self.ConvDiatomComp = nn.ModuleList()
        # DiatomComp input size is batch_size x 30 x 6 but should be batch_size x 6 x 30
        self.ConvDiatomComp.append(nn.Conv1d(6, config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start'], config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.MaxPool1d(config['DiatomComp_MaxPool_kernel_size'], config['DiatomComp_MaxPool_kernel_size']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start'], config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same')) #Input is probably 64 now?
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvDiatomComp.append(nn.Conv1d(config['DiatomComp_out_channels_start']//2, config['DiatomComp_out_channels_start']//2, config['DiatomComp_kernel_size'], padding='same'))
        self.ConvDiatomComp.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        # Flatten
        self.ConvDiatomComp.append(nn.Flatten())

        # Calculate the output size of the DiatomComp layers
        ConvDiAtomCompSize = (30 // config['DiatomComp_MaxPool_kernel_size']) * (config['DiatomComp_out_channels_start']//2)
        print(ConvDiAtomCompSize)

        self.ConvGlobal = nn.ModuleList()
        # Global input size is batch_size x 60
        self.ConvGlobal.append(nn.Linear(60, config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.ConvGlobal.append(nn.Linear(config['Global_units'], config['Global_units']))
        self.ConvGlobal.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))

        # Calculate the output size of the Global layers
        ConvGlobal_output_size = config['Global_units']
        print(ConvGlobal_output_size)

        # One-hot encoding
        self.OneHot = nn.ModuleList()
        self.OneHot.append(nn.Conv1d(20, config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.Conv1d(config['OneHot_out_channels'], config['OneHot_out_channels'], config['One_hot_kernel_size'], padding='same'))
        self.OneHot.append(nn.Tanh())
        self.OneHot.append(nn.MaxPool1d(config['OneHot_MaxPool_kernel_size'], config['OneHot_MaxPool_kernel_size']))
        self.OneHot.append(nn.Flatten())

        # Calculate the output size of the OneHot layers
        conv_output_size_OneHot = ((60 // config['OneHot_MaxPool_kernel_size']) * config['OneHot_out_channels'])
        print(conv_output_size_OneHot)

        # Calculate the total input size for the Concat layer
        total_input_size = ConvAtomCompSize + ConvDiAtomCompSize + ConvGlobal_output_size + conv_output_size_OneHot
        print(total_input_size)

        self.total_input_size = total_input_size

        # Concatenate
        self.Concat = nn.ModuleList()
        self.Concat.append(nn.Linear(total_input_size, config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))
        self.Concat.append(nn.Linear(config['Concat_units'], config['Concat_units']))
        self.Concat.append(LRelu_with_saturation(config['LRelu_negative_slope'], config['LRelu_saturation']))

        self.Concat.append(nn.Linear(config['Concat_units'], 1))

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)
        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)
        for layer in self.OneHot:
            one_hot = layer(one_hot)

        concatenated = torch.cat((atom_comp, diatom_comp, one_hot, global_feats), 1)
        for layer in self.Concat:
            concatenated = layer(concatenated)
        output = concatenated
        return output

class Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fcoutput = nn.Linear(output_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fcoutput(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

class IM2DeepMultiTransfer(L.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepMultiTransfer, self).__init__()
        self.config = config
        self.criterion = criterion
        self.l1_alpha = config['L1_alpha']

        # Load the IM2Deep model
        self.backbone = IM2Deep(BASEMODELCONFIG)
        self.backbone.load_state_dict(torch.load('/home/robbe/IM2DeepMulti/BestParams_final_model_state_dict.pth'))
        self.ConvAtomComp = self.backbone.ConvAtomComp
        self.ConvDiatomComp = self.backbone.ConvDiatomComp
        self.ConvGlobal = self.backbone.ConvGlobal
        self.OneHot = self.backbone.OneHot

        self.concat = list(self.backbone.Concat.children())[:-1]

        self.branches = nn.ModuleList([Branch(94, config['BranchSize']), Branch(94, config['BranchSize'])])
        # self.outputlayer = OutputLayer(94, 2)

        # self.log_sigma_squared1 = nn.Parameter(torch.tensor([0.0]))
        # self.log_sigma_squared2 = nn.Parameter(torch.tensor([0.0]))

    def DeepLCNN_transfer(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)

        for layer in self.ConvAtomComp:
            atom_comp = layer(atom_comp)

        for layer in self.ConvDiatomComp:
            diatom_comp = layer(diatom_comp)

        for layer in self.ConvGlobal:
            global_feats = layer(global_feats)

        for layer in self.OneHot:
            one_hot = layer(one_hot)

        CNNoutput = torch.cat((atom_comp, diatom_comp, global_feats, one_hot), dim=1)

        for layer in self.concat:
            CNNoutput = layer(CNNoutput)
        return CNNoutput

    def training_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)


        # Y_hats are of shape (batch_size, 1) but should be (batch_size, )
        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.l1_alpha * l1_norm

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Train Loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)

        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)

        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        y1, y2 = y[:, 0], y[:, 1]

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)


        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y1, y2, y_hat1, y_hat2)
        meanmae = MeanMAESorted(y1, y2, y_hat1, y_hat2)
        lowestmae = LowestMAESorted(y1, y2, y_hat1, y_hat2)

        self.log('Test Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Test Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)

        return torch.hstack([y_hat1, y_hat2])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer



