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
from models import IM2DeepMulti, IM2DeepMultiTransfer, IM2DeepMultiTransferWithAttention
from prepare_data import prepare_data
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint, ModelSummary, RichProgressBar
from utils import evaluate_predictions, plot_predictions, MultiOutputLoss, PredictionWriter, WeightedLoss, FlexibleLoss, FlexibleLossSorted, FlexibleLossWithDynamicWeight

def main():
    torch.set_float32_matmul_precision('high')
    config = {
        "name": "Sweep",
        "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "batch_size": 32,
        "learning_rate": 0.0000885185503354955,
        "diversity_weight": 1.2631492541307987,    # Should be high when using FlexibleLoss (4.2), much lower when using FlexibleLossSorted (1)
        "L1_alpha": 0.00000043707211872154, #0.00003 for FlexibleLoss, 0.02 for FlexibleLossSorted
        "epochs": 10,
        "delta": 0,
        "device": "1",
        "Use_best_model": True,
        "Add_branch_layer": False,
        'BranchSize': 58, #64 seems to be the best
        'Loss_type': 'FlexibleLoss',
        'Use_attention_output': True,
        'Use_attention_concat': True,
    }

    wandb.init(project="IM2DeepMulti", config=config, name=config["name"] + "-" + config["time"], save_code=False)
    config = wandb.config

    ccs_df_train, train_loader, ccs_df_valid, valid_loader, ccs_df_test, test_loader = (
        prepare_data(config)
    )

    if config['Loss_type'] == 'FlexibleLoss':
        criterion = FlexibleLoss(config['diversity_weight'])
    elif config['Loss_type'] == 'FlexibleLossSorted':
        criterion = FlexibleLossSorted(config['diversity_weight'])
    # criterion = FlexibleLossWithDynamicWeight(config['diversity_weight'])

    if config['Use_attention_output'] or config['Use_attention_concat']:
        model = IM2DeepMultiTransferWithAttention(config, criterion)
    else:
        model = IM2DeepMultiTransfer(config, criterion)

    print(model)

    mcp = ModelCheckpoint(
        dirpath="/home/robbe/IM2DeepMulti/checkpoints/",
        filename=config["name"] + "-" + config["time"],
        monitor="Val Mean MAE",
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
    # Load best model
    if config["Use_best_model"]:
        if config['Use_attention_output'] or config['Use_attention_concat']:
            model = IM2DeepMultiTransferWithAttention.load_from_checkpoint(mcp.best_model_path, config=config, criterion=criterion)
        else:
            model = IM2DeepMultiTransfer.load_from_checkpoint(mcp.best_model_path, config=config, criterion=criterion)

    predictions = trainer.predict(model, test_loader) # Predictions is a list of tensors
    predictions = torch.vstack(predictions).detach().cpu().numpy()
    # Targets now looks is of shape (n_samples, 1) where 1 is an array of len 2 but we need to reshape it to (n_samples, 2)
    targets = ccs_df_test["CCS"].values.reshape(-1, 1)
    targets = np.array([x[0] for x in targets])

    # Evaluate the predictions
    test_mean_mae, test_lowest_mae, test_mean_pearson_r, mean_mre = evaluate_predictions(
        predictions, targets
    )

    wandb.log(
        {
            "Test Mean MAE": test_mean_mae,
            "Test Lowest MAE": test_lowest_mae,
            "Test Mean Pearson R": test_mean_pearson_r,
            "Test Mean MRE": mean_mre,
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
