import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from datetime import datetime

MAE = nn.L1Loss()

class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super(LRelu_with_saturation, self).__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval, config):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.config = config

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "{}-{}-predictions.pt".format(self.config["name"], self.config["time"])))

class MultiOutputLoss(nn.Module):
    def __init__(self, coefficient):
        super(MultiOutputLoss, self).__init__()
        self.coefficient = coefficient

    def forward(self, y1, y2, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()

        if torch.equal(y1, y2):
            weight = 1 - self.coefficient
        else:
            weight = self.coefficient

        loss1 = loss_fn(y_hat1, y1)
        loss2 = loss_fn(y_hat2, y2)

        # TODO: Loss component for capturing the difference between the two outputs
        # deltaloss = torch.mean(abs((y_hat1 - y_hat2) - (y1 - y2)))
        return (loss1 + loss2) * weight

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, y1, y2, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()

        loss1 = loss_fn(y_hat1, y1)
        loss2 = loss_fn(y_hat2, y2)

        # similarity_factor = torch.abs(y1-y2) + 1

        # total_loss = (loss1 + loss2) * similarity_factor
        # total_loss_scalar = torch.mean(total_loss)
        return loss1 + loss2

class FlexibleLoss(nn.Module):
    def __init__(self):
        super(FlexibleLoss, self).__init__()

    def forward(self, y1, y2, y_hat1, y_hat2):
        diversity_weight = 0.4
        loss_fn = nn.L1Loss()

        loss1_to_1 = loss_fn(y_hat1, y1)
        loss2_to_2 = loss_fn(y_hat2, y2)
        loss1_to_2 = loss_fn(y_hat1, y2)
        loss2_to_1 = loss_fn(y_hat2, y1)

        loss_dict = {"1_to_1": loss1_to_1, "2_to_2": loss2_to_2, "1_to_2": loss1_to_2, "2_to_1": loss2_to_1}
        min_loss_key = min(loss_dict, key=loss_dict.get)
        if "1_to" in min_loss_key:
            if "to_1" in min_loss_key:
                loss1 = loss1_to_1
                loss2 = loss2_to_2
            else:
                loss1 = loss1_to_2
                loss2 = loss2_to_1
        else:
            if "to_2" in min_loss_key:
                loss1 = loss2_to_2
                loss2 = loss1_to_1
            else:
                loss1 = loss2_to_1
                loss2 = loss1_to_2

        diversity_term = torch.mean(torch.abs(y_hat1 - y_hat2))
        total_loss = (loss1 + loss2) - (diversity_weight * diversity_term)

        return total_loss

class FlexibleLossSorted(nn.Module):
    def __init__(self):
        super(FlexibleLossSorted, self).__init__()

    def forward(self, y1, y2, y_hat1, y_hat2):
        diversity_weight = 0.4
        loss_fn = nn.L1Loss()

        # Sort the targets and predictions row-wise
        targets = torch.stack([y1, y2], dim=1)
        predictions = torch.stack([y_hat1, y_hat2], dim=1)
        targets, _ = torch.sort(targets, dim=1)
        predictions, _ = torch.sort(predictions, dim=1)

        target1 = targets[:, 0]
        target2 = targets[:, 1]

        prediction1 = predictions[:, 0]
        prediction2 = predictions[:, 1]

        loss1 = loss_fn(prediction1, target1)
        loss2 = loss_fn(prediction2, target2)

        diversity_term = torch.mean(torch.abs(prediction1 - prediction2))
        total_loss = (loss1 + loss2) - (diversity_weight * diversity_term)

        return total_loss

def MeanMAE(y1, y2, y_hat1, y_hat2):
    mae1_to_1 = MAE(y_hat1, y1)
    mae2_to_2 = MAE(y_hat2, y2)
    mae1_to_2 = MAE(y_hat1, y2)
    mae2_to_1 = MAE(y_hat2, y1)

    mae_dict = {"1_to_1": mae1_to_1, "2_to_2": mae2_to_2, "1_to_2": mae1_to_2, "2_to_1": mae2_to_1}
    min_mae_key = min(mae_dict, key=mae_dict.get)
    if "1_to" in min_mae_key:
        if "to_1" in min_mae_key:
            mae1 = mae1_to_1
            mae2 = mae2_to_2
        else:
            mae1 = mae1_to_2
            mae2 = mae2_to_1
    else:
        if "to_2" in min_mae_key:
            mae1 = mae2_to_2
            mae2 = mae1_to_1
        else:
            mae1 = mae2_to_1
            mae2 = mae1_to_2

    return (mae1 + mae2) / 2

def MeanMAESorted(y1, y2, y_hat1, y_hat2):
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)
    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    mae1 = MAE(prediction1, target1)
    mae2 = MAE(prediction2, target2)

    return (mae1 + mae2) / 2

def LowestMAE(y1, y2, y_hat1, y_hat2):
    mae1_to_1 = MAE(y_hat1, y1)
    mae2_to_2 = MAE(y_hat2, y2)
    mae1_to_2 = MAE(y_hat1, y2)
    mae2_to_1 = MAE(y_hat2, y1)

    mae_dict = {"1_to_1": mae1_to_1, "2_to_2": mae2_to_2, "1_to_2": mae1_to_2, "2_to_1": mae2_to_1}
    min_mae_key = min(mae_dict, key=mae_dict.get)
    if "1_to" in min_mae_key:
        if "to_1" in min_mae_key:
            mae1 = mae1_to_1
            mae2 = mae2_to_2
        else:
            mae1 = mae1_to_2
            mae2 = mae2_to_1
    else:
        if "to_2" in min_mae_key:
            mae1 = mae2_to_2
            mae2 = mae1_to_1
        else:
            mae1 = mae2_to_1
            mae2 = mae1_to_2

    return min(mae1, mae2)

def LowestMAESorted(y1, y2, y_hat1, y_hat2):
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)

    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    mae1 = MAE(prediction1, target1)
    mae2 = MAE(prediction2, target2)

    return min(mae1, mae2)

def MeanPearsonR(y1, y2, y_hat1, y_hat2):
    r1 = pearsonr(y1, y_hat1)[0]
    r2 = pearsonr(y2, y_hat2)[0]
    return (r1 + r2) / 2

def MeanPearsonRSorted(y1, y2, y_hat1, y_hat2):
    targets = torch.stack([y1, y2], dim=1)
    predictions = torch.stack([y_hat1, y_hat2], dim=1)

    targets, _ = torch.sort(targets, dim=1)
    predictions, _ = torch.sort(predictions, dim=1)

    target1 = targets[:, 0]
    target2 = targets[:, 1]

    prediction1 = predictions[:, 0]
    prediction2 = predictions[:, 1]

    r1 = pearsonr(target1, prediction1)[0]
    r2 = pearsonr(target2, prediction2)[0]

    return (r1 + r2) / 2

BASEMODELCONFIG = {
        "AtomComp_kernel_size": 4,
        "DiatomComp_kernel_size": 4,
        "One_hot_kernel_size": 4,
        "AtomComp_out_channels_start": 356,
        "DiatomComp_out_channels_start": 65,
        "Global_units": 20,
        "OneHot_out_channels": 1,
        "Concat_units": 94,
        "AtomComp_MaxPool_kernel_size": 2,
        "DiatomComp_MaxPool_kernel_size": 2,
        "OneHot_MaxPool_kernel_size": 10,
        "LRelu_negative_slope": 0.013545684190756122,
        "LRelu_saturation": 40,
        }

def evaluate_predictions(predictions, targets):
    predictions = np.sort(predictions, axis=1)
    targets = np.sort(targets, axis=1)
    predictions1 = torch.tensor(predictions[:, 0])
    predictions2 = torch.tensor(predictions[:, 1])
    targets1 = torch.tensor(targets[:, 0])
    targets2 = torch.tensor(targets[:, 1])

    mean_mae = MeanMAESorted(targets1, targets2, predictions1, predictions2)
    lowest_mae = LowestMAESorted(targets1, targets2, predictions1, predictions2)

    mean_pearson_r = MeanPearsonRSorted(targets1, targets2, predictions1, predictions2)

    return mean_mae, lowest_mae, mean_pearson_r

def plot_predictions(predictions, targets, mean_mae, mean_pearson_r, config):
    # Sort prediction array row-wise
    predictions = np.sort(predictions, axis=1)
    # Sort target array row-wise
    targets = np.sort(targets, axis=1)
    predictions1 = predictions[:, 0]
    predictions2 = predictions[:, 1]
    targets1 = targets[:, 0]
    targets2 = targets[:, 1]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].scatter(targets1, predictions1, label="Predictions 1", s=1)
    axes[1].scatter(targets2, predictions2, label="Predictions 2", s=1)
    axes[0].plot([min(targets1), max(targets1)], [min(targets1), max(targets1)], color="red")
    axes[1].plot([min(targets2), max(targets2)], [min(targets2), max(targets2)], color="red")
    axes[0].set_xlabel("Observed CCS")
    axes[0].set_ylabel("Predicted CCS")
    axes[1].set_xlabel("Observed CCS")
    axes[1].set_ylabel("Predicted CCS")
    plt.suptitle(f"Mean MAE: {mean_mae}, Mean Pearson R: {mean_pearson_r}")
    plt.savefig("/home/robbe/IM2DeepMulti/figs/{}-{}.png".format(config["name"], config["time"]))

    DeltaPred = abs(predictions1 - predictions2)
    DeltaCCS = abs(targets1 - targets2)

    fig = plt.figure(figsize=(8,8))
    plt.scatter(DeltaCCS, DeltaPred, s=3)
    plt.plot([min(DeltaCCS), max(DeltaCCS)], [min(DeltaCCS), max(DeltaCCS)], color="red")
    plt.xlabel('Difference between observed CCS')
    plt.ylabel('Difference between predicted CCS')
    plt.savefig("/home/robbe/IM2DeepMulti/figs/Deltas-{}-{}.png".format(config['name'], config['time']))
    plt.close()


