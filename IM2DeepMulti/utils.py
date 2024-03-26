import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from lightning.pytorch.callbacks import BasePredictionWriter

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
    def __init__(self):
        super(MultiOutputLoss, self).__init__()

    def forward(self, y1, y2, y_hat1, y_hat2):
        loss_fn = nn.L1Loss()

        loss1 = loss_fn(y_hat1, y1)
        loss2 = loss_fn(y_hat2, y2)

        # TODO: Loss component for capturing the difference between the two outputs

        return (loss1 + loss2)

def MeanMAE(y1, y2, y_hat1, y_hat2):
    mae1 = MAE(y1, y_hat1)
    mae2 = MAE(y2, y_hat2)
    return (mae1 + mae2) / 2

def LowestMAE(y1, y2, y_hat1, y_hat2):
    mae1 = MAE(y1, y_hat1)
    mae2 = MAE(y2, y_hat2)
    return min(mae1, mae2)

def MeanPearsonR(y1, y2, y_hat1, y_hat2):
    r1 = pearsonr(y1, y_hat1)[0]
    r2 = pearsonr(y2, y_hat2)[0]
    return (r1 + r2) / 2

def evaluate_predictions(predictions, targets):
    predictions1 = torch.tensor(predictions[:, 0])
    predictions2 = torch.tensor(predictions[:, 1])
    targets1 = torch.tensor(targets[:, 0])
    targets2 = torch.tensor(targets[:, 1])

    mean_mae = MeanMAE(targets1, targets2, predictions1, predictions2)
    lowest_mae = LowestMAE(targets1, targets2, predictions1, predictions2)

    mean_pearson_r = MeanPearsonR(targets1, targets2, predictions1, predictions2)

    return mean_mae, lowest_mae, mean_pearson_r

def plot_predictions(predictions, targets, mean_mae, mean_pearson_r, config):
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

