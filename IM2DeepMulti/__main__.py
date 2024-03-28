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
import lightning as L
from models import IM2DeepMulti
from prepare_data import prepare_data
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint, ModelSummary, RichProgressBar
from utils import evaluate_predictions, plot_predictions, MultiOutputLoss, PredictionWriter, WeightedLoss, FlexibleLoss

def main():
    torch.set_float32_matmul_precision('high')
    config = {
        "name": "OnlyMultimodals-SymmetricWeightedLoss",
        "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "batch_size": 32,
        "learning_rate": 0.0001,
        "AtomComp_kernel_size": 4,
        "DiatomComp_kernel_size": 4,
        "One_hot_kernel_size": 4,
        "AtomComp_out_channels_start": 256,
        "DiatomComp_out_channels_start": 64,
        "Global_units": 10,
        "OneHot_out_channels": 1,
        "Concat_units": 49,
        "AtomComp_MaxPool_kernel_size": 2,
        "DiatomComp_MaxPool_kernel_size": 2,
        "OneHot_MaxPool_kernel_size": 10,
        "LRelu_negative_slope": 0.013545684190756122,
        "LRelu_saturation": 40,
        "L1_alpha": 0.000006056805819927765,
        "epochs": 300,
        "delta": 0,
        "device": "0",
        "Use_best_model": False,
    }

    wandb.init(project="IM2DeepMulti", config=config, name=config["name"] + "-" + config["time"], save_code=False)
    config = wandb.config

    ccs_df_train, train_loader, ccs_df_valid, valid_loader, ccs_df_test, test_loader = (
        prepare_data(config)
    )

    # criterion = MultiOutputLoss(coefficient=0.8)
    criterion = FlexibleLoss()

    model = IM2DeepMulti(config, criterion)
    # pred_writer = PredictionWriter(write_interval="epoch", output_dir="/home/robbe/IM2DeepMulti/preds", config=config)
    mcp = ModelCheckpoint(
        filename=config["name"] + "-" + config["time"],
        monitor="Val Loss",
        mode="min",
        save_last=False,
        verbose=False,
    )
    wandb_logger = WandbLogger(project="IM2DeepMulti", log_model=True, save_dir='/home/robbe/IM2DeepMulti/checkpoints')
    wandb_logger.watch(model)

    trainer = L.Trainer(
        devices=1,
        accelerator="auto",
        enable_checkpointing=True,
        max_epochs=config["epochs"],
        enable_progress_bar=True,
        callbacks=[mcp, ModelSummary(), RichProgressBar()],
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    # prediction1, prediction2 = trainer.predict(model, test_loader)
    predictions = trainer.predict(model, test_loader) # Predictions is a list of tensors
    predictions = torch.vstack(predictions).detach().cpu().numpy()
    # Targets now looks is of shape (n_samples, 1) where 1 is an array of len 2 but we need to reshape it to (n_samples, 2)
    targets = ccs_df_test["CCS"].values.reshape(-1, 1)
    targets = np.array([x[0] for x in targets])

    # Evaluate the predictions
    test_mean_mae, test_lowest_mae, test_mean_pearson_r = evaluate_predictions(
        predictions, targets
    )

    wandb.log(
        {
            "Test Mean MAE": test_mean_mae,
            "Test Lowest MAE": test_lowest_mae,
            "Test Mean Pearson R": test_mean_pearson_r,
        }
    )

    # Plot the predictions
    plot_predictions(predictions, targets, test_mean_mae, test_mean_pearson_r, config)

    # Add the predictions to the test_df
    ccs_df_test["predicted_CCS1"] = predictions[:, 0]
    ccs_df_test["predicted_CCS2"] = predictions[:, 1]
    ccs_df_test.to_csv("/home/robbe/IM2DeepMulti/preds/output/Test-{}-{}.csv".format(config['name'], config['time']), index=False)
    ccs_df_test.to_pickle("/home/robbe/IM2DeepMulti/preds/Test-{}-{}.pkl".format(config['name'], config['time']))

if __name__ == "__main__":
    # os.environ["WANDB_MODE"] = "dryrun"
    main()
