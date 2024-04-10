import sys
import os
sys.path.append('/home/robbe/IM2DeepMulti')
sys.path.append('/home/robbe/IM2DeepMulti/IM2DeepMulti')
import torch
import torch.nn as nn
from IM2DeepMulti.models import IM2Deep, SelfAttention, Branch
from IM2DeepMulti.utils import BASEMODELCONFIG, FlexibleLossSorted, evaluate_predictions, plot_predictions, LowestMAESorted, MeanMAESorted
from datetime import datetime
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary, RichProgressBar
import wandb
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class DecoyLoss(nn.Module):
    def __init__(self):
        super(DecoyLoss, self).__init__()

    def forward(self, y, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()
        loss1 = loss_fn(y, y_hat1)
        loss2 = loss_fn(y, y_hat2)
        return loss1 + loss2

def MeanMAE(y1, y_hat1, y_hat2):
    return (torch.mean(torch.abs(y1 - y_hat1)) + torch.mean(torch.abs(y1 - y_hat2))) / 2

def LowestMAE(y1, y_hat1, y_hat2):
    return min(torch.mean(torch.abs(y1 - y_hat1)), torch.mean(torch.abs(y1 - y_hat2)))

def MeanPearsonR(y1, y_hat1, y_hat2):
    return (torch.mean(torch.abs(stats.pearsonr(y1, y_hat1))) + torch.mean(torch.abs(stats.pearsonr(y1, y_hat2))) / 2)

def MeanMRE(y1, y_hat1, y_hat2):
    return (torch.mean(torch.abs((y1 - y_hat1) / y1)) + torch.mean(torch.abs((y1 - y_hat2) / y1))) / 2

class IM2DeepMultiTransferWithAttentionDecoy(pl.LightningModule):
    def __init__(self, config, criterion):
        super(IM2DeepMultiTransferWithAttentionDecoy, self).__init__()
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
        self.SelfAttentionConcat = SelfAttention(1841, 1)
        self.SelfAttentionOutput = SelfAttention(94, 1)

        self.branches = nn.ModuleList([Branch(94, config['BranchSize'], config['Add_branch_layer']), Branch(94, config['BranchSize'], config['Add_branch_layer'])])

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

        if self.config['Use_attention_concat']:
            CNNoutput = self.SelfAttentionConcat(CNNoutput.unsqueeze(1)).squeeze(1)

        for layer in self.concat:
            CNNoutput = layer(CNNoutput)

        if self.config['Use_attention_output']:
            CNNoutput = self.SelfAttentionOutput(CNNoutput.unsqueeze(1)).squeeze(1)
        return CNNoutput

    def training_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)

        # Y_hats are of shape (batch_size, 1) but should be (batch_size, )
        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y, y_hat1, y_hat2)

        l1_norm = sum(p.abs().sum() for p in self.parameters())
        total_loss = loss + self.l1_alpha * l1_norm

        meanmae = MeanMAE(y, y_hat1, y_hat2)
        lowestmae = LowestMAE(y, y_hat1, y_hat2)

        self.log('Train Loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Train Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch

        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)
        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)

        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = self.criterion(y, y_hat1, y_hat2)

        meanmae = MeanMAE(y, y_hat1, y_hat2)
        lowestmae = LowestMAE(y, y_hat1, y_hat2)

        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Mean MAE', meanmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Val Lowest MAE', lowestmae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        AtomEnc, DiAtomEnc, Globals, OneHot, y = batch
        test_loss = FlexibleLossSorted(self.config['diversity_weight'])
        y1, y2 = y[:, 0], y[:, 1]
        CNN_output = self.DeepLCNN_transfer(AtomEnc, DiAtomEnc, Globals, OneHot)
        # y_hat1, y_hat2 = self.outputlayer(CNN_output).split(1, dim=1)


        y_hat1 = self.branches[0](CNN_output)
        y_hat2 = self.branches[1](CNN_output)


        y_hat1 = y_hat1.squeeze(1)
        y_hat2 = y_hat2.squeeze(1)

        loss = test_loss(y1, y2, y_hat1, y_hat2)
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

def main():
    torch.set_float32_matmul_precision('high')
    config = {
            "name": "DecoyModelMultiRandom",
            "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "batch_size": 32,
            "learning_rate": 0.0000885185503354955,
            "diversity_weight": 1.2631492541307987,
            "L1_alpha": 0.00000043707211872154,
            "epochs": 300,
            "delta": 0,
            "device": "1",
            "Use_best_model": True,
            "Add_branch_layer": False,
            'BranchSize': 58,
            'Loss_type': 'FlexibleLoss',
            'Use_attention_output': True,
            'Use_attention_concat': True,
        }

    criterion = DecoyLoss()
    decoy_model = IM2DeepMultiTransferWithAttentionDecoy(config, criterion)

    ccs_df_train = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_train_OnlyMoreMultimodals.pkl', 'rb')).reset_index(drop=True)
    AtomEnc = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_train_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Globals = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_train_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb"))
    DiAminoAtomEnc = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_train_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    OneHot = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_train_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_train = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_train-OnlyMoreMultimodals.pickle', 'rb'))

    ccs_df_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_valid_OnlyMoreMultimodals.pkl', 'rb')).reset_index(drop=True)
    AtomEnc_valid = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_valid_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Globals_valid = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_valid_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb"))
    DiAminoAtomEnc_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_valid_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    OneHot_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_valid_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_valid = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_valid-OnlyMoreMultimodals.pickle', 'rb'))

    ccs_df_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/ccs_df_test_OnlyMoreMultimodals.pkl', 'rb')).reset_index(drop=True)
    AtomEnc_test = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_test_AtomEnc-OnlyMoreMultimodals.pickle", "rb"))
    Globals_test = pickle.load(open("/home/robbe/IM2DeepMulti/data/X_test_GlobalFeatures-OnlyMoreMultimodals.pickle", "rb"))
    DiAminoAtomEnc_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_test_DiAminoAtomEnc-OnlyMoreMultimodals.pickle', 'rb'))
    OneHot_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/X_test_OneHot-OnlyMoreMultimodals.pickle', 'rb'))
    y_test = pickle.load(open('/home/robbe/IM2DeepMulti/data/y_test-OnlyMoreMultimodals.pickle', 'rb'))

    # Convert the data to PyTorch tensors
    AtomEnc_train = torch.tensor(AtomEnc, dtype=torch.float32)
    Globals_train = torch.tensor(Globals, dtype=torch.float32)
    DiAminoAtomEnc_train = torch.tensor(DiAminoAtomEnc, dtype=torch.float32)
    OneHot_train = torch.tensor(OneHot, dtype=torch.float32)
    y_train = torch.tensor(np.vstack(y_train))
    # Randomly pick one of the two targets as the target for each sample, randommize per sample
    random_indices = torch.randint(0, 2, (y_train.shape[0],))
    y_train = torch.gather(y_train, 1, random_indices.unsqueeze(1)).squeeze(1)

    AtomEnc_valid = torch.tensor(AtomEnc_valid, dtype=torch.float32)
    Globals_valid = torch.tensor(Globals_valid, dtype=torch.float32)
    DiAminoAtomEnc_valid = torch.tensor(DiAminoAtomEnc_valid, dtype=torch.float32)
    OneHot_valid = torch.tensor(OneHot_valid, dtype=torch.float32)
    y_valid = torch.tensor(np.vstack(y_valid))
    print(y_valid.shape)
    random_indices = torch.randint(0, 2, (y_valid.shape[0],))
    y_valid = torch.gather(y_valid, 1, random_indices.unsqueeze(1)).squeeze(1)
    print(y_valid.shape)

    AtomEnc_test = torch.tensor(AtomEnc_test, dtype=torch.float32)
    Globals_test = torch.tensor(Globals_test, dtype=torch.float32)
    DiAminoAtomEnc_test = torch.tensor(DiAminoAtomEnc_test, dtype=torch.float32)
    OneHot_test = torch.tensor(OneHot_test, dtype=torch.float32)
    y_test = torch.tensor(np.vstack(y_test)).squeeze(1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(AtomEnc_train, DiAminoAtomEnc_train, Globals_train, OneHot_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(AtomEnc_valid, DiAminoAtomEnc_valid, Globals_valid, OneHot_valid, y_valid)
    test_dataset = torch.utils.data.TensorDataset(AtomEnc_test, DiAminoAtomEnc_test, Globals_test, OneHot_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=31)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=31)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=31)

    mcp = ModelCheckpoint(
        dirpath="/home/robbe/IM2DeepMulti/checkpoints/decoyeval/",
        filename=config["name"] + "-" + config["time"],
        monitor="Val Mean MAE",
        mode="min",
        save_last=False,
        verbose=False,
    )
    wandb_logger = WandbLogger(project="IM2DeepMulti", log_model=True, save_dir='/home/robbe/IM2DeepMulti/checkpoints/decoyeval/')
    wandb_logger.watch(decoy_model)

    trainer = pl.Trainer(
        devices=[int(config["device"])],
        accelerator="gpu",
        enable_checkpointing=True,
        max_epochs=config["epochs"],
        enable_progress_bar=True,
        callbacks=[mcp, ModelSummary(), RichProgressBar()],
        logger=wandb_logger,
    )

    trainer.fit(decoy_model, train_loader, valid_loader)
    trainer.test(decoy_model, test_loader)

    # Load best model
    decoy_model = IM2DeepMultiTransferWithAttentionDecoy.load_from_checkpoint(mcp.best_model_path, config=config, criterion=criterion)

    predictions = trainer.predict(decoy_model, test_loader) # Predictions is a list of tensors
    predictions = torch.vstack(predictions).detach().cpu().numpy()
    targets = ccs_df_test['CCS'].values.reshape(-1, 1)
    targets = np.array([x[0] for x in targets])


    mean_mae, lowest_mae, mean_pearson_r, mean_mre = evaluate_predictions(predictions, targets)

    wandb.log(
        {
            "Test Mean MAE": mean_mae,
            "Test Lowest MAE": lowest_mae,
            "Test Mean Pearson R": mean_pearson_r,
            "Test Mean MRE": mean_mre,
        }
    )

    plot_predictions(predictions, targets, mean_mae, mean_pearson_r, config, '/home/robbe/IM2DeepMulti/figs/decoyeval/')

    ccs_df_test['DecoyModel_Prediction1'] = predictions[:, 0]
    ccs_df_test['DecoyModel_Prediction2'] = predictions[:, 1]
    ccs_df_test.to_csv(f'/home/robbe/IM2DeepMulti/preds/decoyeval/{config["name"]}-{config["time"]}.csv')
    ccs_df_test.to_pickle(f'/home/robbe/IM2DeepMulti/preds/decoyeval/{config["name"]}-{config["time"]}.pkl')

if __name__ == '__main__':
    os.environ["WANDB_MODE"] = "dryrun"
    main()








