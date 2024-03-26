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

class SumLoss(nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()

    def forward(self, outputs, targets):
        targets = targets[:,0]
        loss_fn = nn.L1Loss()
        prediction1 = outputs[:, 0]
        prediction2 = outputs[:, 1]
        target1 = targets[:, 0]
        # print("target1:", target1)
        target2 = targets[:, 1]
        loss1 = loss_fn(prediction1, target1)
        loss2 = loss_fn(prediction2, target2)
        total_loss = loss1 + loss2
        return total_loss, loss1, loss2

class DeltaLoss(nn.Module):
    def __init__(self):
        super(DeltaLoss, self).__init__()

    def forward(self, outputs, targets):
        loss_fn = nn.L1Loss()

        targets = targets[:,0]

        prediction1 = outputs[:, 0]
        prediction2 = outputs[:, 1]
        target1 = targets[:, 0]
        target2 = targets[:, 1]
        deltatarget = target1 - target2
        deltaprediction = prediction1 - prediction2

        deltaloss = loss_fn(deltaprediction, deltatarget)
        loss1 = loss_fn(prediction1, target1)
        loss2 = loss_fn(prediction2, target2)

        delta_weight = 100
        accuracy_weight = 1
        wandb.log({'delta_weight': delta_weight, 'accuracy_weight': accuracy_weight})


        accuracy_loss = loss1 + loss2
        total_loss = accuracy_weight * accuracy_loss + delta_weight * deltaloss
        return total_loss, loss1, loss2

def compute_sample_weights(targets):
    target1 = targets[:, 0]
    target2 = targets[:, 1]
    deltatarget = target1 - target2
    sample_weights = np.abs(deltatarget)
    return sample_weights

class LossWithSampleWeights(nn.Module):
    def __init__(self):
        super(LossWithSampleWeights, self).__init__()

    def forward(self, outputs, targets, sample_weights):
        loss_fn = nn.L1Loss()
        targets = targets[:,0]
        prediction1 = outputs[:, 0]
        prediction2 = outputs[:, 1]
        target1 = targets[:, 0]
        target2 = targets[:, 1]
        deltatarget = target1 - target2
        deltaprediction = prediction1 - prediction2

        deltaloss = loss_fn(deltaprediction, deltatarget)
        loss1 = loss_fn(prediction1, target1)
        loss2 = loss_fn(prediction2, target2)
        # print(loss1)
        # print(loss2)

        delta_weight = 100
        accuracy_weight = 1
        wandb.log({'delta_weight': delta_weight, 'accuracy_weight': accuracy_weight})


        accuracy_loss = loss1 + loss2
        total_loss = (accuracy_weight * accuracy_loss + delta_weight * deltaloss) * sample_weights.mean()
        return total_loss, loss1, loss2


class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super(LRelu_with_saturation, self).__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)

class DeepLC_mimic(nn.Module):
    def __init__(self, config):
        super(DeepLC_mimic, self).__init__()
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

        self.Concat.append(nn.Linear(config['Concat_units'], 2))

    def forward(self, atom_comp, diatom_comp, global_feats, one_hot):
        atom_comp = atom_comp.permute(0, 2, 1)
        diatom_comp = diatom_comp.permute(0, 2, 1)
        one_hot = one_hot.permute(0, 2, 1)
        # print("Before forward")
        # print(atom_comp.shape)
        # print(diatom_comp.shape)
        # print(global_feats.shape)
        # print(one_hot.shape)

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
        outputs = concatenated
        return outputs

def mean_absolute_error(targets, predictions):
    """
    Calculate the mean absolute error (MAE).
    """
    # Sort targets and predictions
    targets = np.array([sorted(x) for x in targets])
    predictions = np.array([sorted(x) for x in predictions])
    target1, target2 = targets[:, 0], targets[:, 1]
    prediction1, prediction2 = predictions[:, 0], predictions[:, 1]
    mae11 = np.mean(np.abs(target1 - prediction1))
    mae22 = np.mean(np.abs(target2 - prediction2))
    mae12 = np.mean(np.abs(target1 - prediction2))
    mae21 = np.mean(np.abs(target2 - prediction1))
    return min([mae11, mae22, mae12, mae21]), max([mae11, mae22, mae12, mae21]), np.mean([mae11, mae22, mae12, mae21])

def train_model(
    model, criterion, optimizer, train_loader, valid_loader, config, num_epochs=100, device="cuda:1"):
    lowest_mae_values = []
    highest_mae_values = []
    mae_values = []
    val_lowest_mae_values = []
    val_highest_mae_values = []
    val_mae_values = []
    best_loss = np.Inf
    best_mae = np.Inf
    best_val_loss = np.Inf
    best_val_mae = np.Inf

    best_model_epochs = []
    for epoch in range(num_epochs):
        start_time = datetime.now()
        model.train()  # Set the model to training mode

        for AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch, targets_batch in train_loader:
            # wandb.log({"current_learning_rate" : scheduler.get_last_lr()[0]})
            optimizer.zero_grad()

            # Forward pass
            outputs = model(AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch)

            # Compute the sample weights
            sample_weights = compute_sample_weights(targets_batch.detach().cpu().numpy())
            sample_weights = torch.tensor(sample_weights, dtype=torch.float32).to(device)
            # Compute the loss
            total_loss, loss1, loss2 = criterion(outputs, targets_batch.unsqueeze(1), sample_weights)

            # L1 regularization
            l1_regularization = torch.tensor(0.0, requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name and ('Concat' in name or 'Conv' in name) and 'OneHot' not in name:
                    l1_regularization += torch.norm(param, 1)
            total_loss += config['L1_alpha'] * l1_regularization


            lowest_mae, highest_mae, mae = mean_absolute_error(targets_batch.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            wandb.log({"step/total_loss": total_loss})
            wandb.log({"step/loss1": loss1})
            wandb.log({"step/loss2": loss2})
            wandb.log({"step/lowest_mae": lowest_mae})
            wandb.log({"step/highest_mae": highest_mae})
            wandb.log({"step/mean_mae": mae})
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

        # Validate the model after each epoch
        validation_total_loss = validate_model(model, criterion, valid_loader, device)
        with torch.no_grad():
            train_predictions = []
            train_targets = []
            val_predictions = []
            val_targets = []
            for AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch, targets_batch in train_loader:
                outputs = model(AtomEnc_batch, DiAminoAtomEnc_batch, Globals_batch, OneHot_batch)
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(targets_batch.cpu().numpy())

            for AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val, targets_batch_val in valid_loader:
                outputs = model(AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets_batch_val.cpu().numpy())

            lowest_mae, highest_mae, mae = mean_absolute_error(
                np.array(train_targets), np.array(train_predictions).squeeze()
            )
            lowest_mae_values.append(lowest_mae)
            highest_mae_values.append(highest_mae)
            mae_values.append(mae)

            val_lowest_mae, val_highest_mae, val_mae = mean_absolute_error(
                np.array(val_targets), np.array(val_predictions).squeeze()
            )
            val_lowest_mae_values.append(val_lowest_mae)
            val_highest_mae_values.append(val_highest_mae)
            val_mae_values.append(val_mae)

        if total_loss < best_loss:
            best_loss = total_loss
        if mae < best_mae:
            best_mae = mae
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        if validation_total_loss < best_val_loss:
            if (validation_total_loss < (best_val_loss - (config['delta'] * best_val_loss))) and config['Use_best_model']:
                print('Saving best model')
                # best_model = copy.deepcopy(model)
                # best_model = model
                best_model_epochs.append(epoch + 1)
                torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'models/{}_{}_best_model.pth'.format(config['name'], config['time']))
            best_val_loss = validation_total_loss

        finish_time = datetime.now()
        train_time = finish_time - start_time

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": total_loss,
            "Mean absolute error (average)": mae,
            "Lowest mean absolute error": lowest_mae,
            "Highest mean absolute error": highest_mae,
            "Validation loss": validation_total_loss,
            "Validation mean absolute error (average)": val_mae,
            "Validation lowest mean absolute error": val_lowest_mae,
            "Validation highest mean absolute error": val_highest_mae,
            "Best Loss": best_loss,
            "Best mean absolute error": best_mae,
            "Best validation loss": best_val_loss,
            "Best validation mean absolute error": best_val_mae,
            "Training time": train_time.total_seconds(),
            "Best model epochs": best_model_epochs,
        })
        print(
            f"Epoch [{epoch+1}/{num_epochs}]: Loss: {total_loss:.4f}, MAE: {mae:.4f}, MAE (lowest): {lowest_mae:.4f}, Validation Loss: {validation_total_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation MAE (lowest): {val_lowest_mae:.4f}, Training time: {train_time.total_seconds()} seconds, Learning rate: {optimizer.param_groups[0]['lr']}"
        )

    if not config['Use_best_model']:
        torch.save({'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, 'models/{}_{}_final_model.pth'.format(config['name'], config['time']))

    print("Training finished!")
    print("Model was saved on these epochs:", best_model_epochs)

    # Save predictions
    train_df = pd.DataFrame({"CCS": train_targets, "Predictions": train_predictions})
    valid_df = pd.DataFrame({"CCS": val_targets, "Predictions": val_predictions})
    train_df.to_pickle(
        "/home/robbe/multioutput_prediction/preds/{}_{}_train_preds.pkl".format(
            config["name"], config["time"]
        )
    )
    valid_df.to_pickle(
        "/home/robbe/multioutput_prediction/preds/{}_{}_valid_preds.pkl".format(
            config["name"], config["time"]
        )
    )
    return model


def validate_model(model, criterion, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_total_loss = 0.0

    with torch.no_grad():
        for AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val, targets_batch_val in valid_loader:
            # Forward pass
            outputs = model(AtomEnc_batch_val, DiAminoAtomEnc_batch_val, Globals_batch_val, OneHot_batch_val)

            # Compute the sample weights
            sample_weights = compute_sample_weights(targets_batch_val.detach().cpu().numpy())
            sample_weights = torch.tensor(sample_weights, dtype=torch.float32).to(device)

            # Compute the loss
            total_loss, loss1, loss2 = criterion(outputs, targets_batch_val.unsqueeze(1), sample_weights=1)

            total_total_loss += total_loss.item()

    # Calculate average validation loss
    avg_total_loss = total_total_loss / len(valid_loader)

    return avg_total_loss

def evaluate_model(model, test_loader, config, test_df, info='', path='/home/robbe/multioutput_prediction/preds/'):
    model.eval()  # Set the model to evaluation mode
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for Test_AtomEnc_batch, Test_DiAminoAtomEnc_batch, Test_Globals_batch, Test_OneHot_batch, y_test_batch in test_loader:
            # Forward pass
            outputs = model(Test_AtomEnc_batch, Test_DiAminoAtomEnc_batch, Test_Globals_batch, Test_OneHot_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(y_test_batch.cpu().numpy())

    lowest_mae, highest_mae, mae = mean_absolute_error(np.array(test_targets), np.array(test_predictions).squeeze())
    mre = np.median(
        abs(np.array(test_predictions).flatten() - np.array(test_targets).flatten())
        / np.array(test_targets).flatten()
    )
    print(f"Test MAE (average): {mae:.4f}")
    print(f"Test MAE (lowest): {lowest_mae:.4f}")
    print(f"Test MAE (highest): {highest_mae:.4f}")

    r, p = stats.pearsonr(
        np.array(test_targets).flatten(), np.array(test_predictions).flatten()
    )
    print(f"Test Pearson R: {r:.4f}")
    perc_95 = round(
        np.percentile(
            (
                abs(
                    np.array(test_predictions).flatten()
                    - np.array(test_targets).flatten()
                )
                / np.array(test_targets).flatten()
            )
            * 100,
            95,
        ),
        2,
    )

    wandb.log({"Test MAE (average)": mae, "Test MAE (lowest)": lowest_mae, "Test MAE (highest)": highest_mae, "Test Pearson R": r, "Test 95th percentile": perc_95, "Test MRE": mre})
    # Save the predictions for each sample #TODO: Give this to the df of the samples
    test_df['Predictions'] = test_predictions
    test_df.to_pickle(
        path + "{}_preds.pkl".format(
            info + "-" + config["name"] + "-" + config["time"]
        )
    )
    return mae, r, perc_95, test_df

def plot_predictions(
    test_df, config, mae, r, perc_95, info='', path='/home/robbe/multioutput_prediction/figs/',
):  # TODO: make charge state a parameter to color
    if len(test_df) < 1e4:
        set_alpha = 0.2
        set_size = 3
    else:
        set_alpha = 0.05
        set_size = 1

    # Plot two scatter plots, one for each target
    print(np.vstack(test_df["CCS"]).shape)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        ax.scatter(
            np.vstack(test_df["CCS"])[:, i],
            np.vstack(test_df["Predictions"])[:, i],
            alpha=set_alpha,
            s=set_size,
        )
        ax.plot([300, 1100], [300, 1100], c="grey")
        ax.set_xlabel("Observed CCS (Å^2)")
        ax.set_ylabel("Predicted CCS (Å^2)")
    fig.suptitle(f"PCC: {round(r, 4)} - MAE: {round(mae, 4)} - 95th percentile: {round(perc_95, 4)}%")
    plt.savefig(
        path + "{}_preds.png".format(
            info + "-" + config["name"] + "-" + config["time"]
        )
    )
    plt.close()

def main():

    config = {
        "name": "Test",
        "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "batch_size": 64,
        "learning_rate": 0.0001,
        "AtomComp_kernel_size": 4,
        "DiatomComp_kernel_size": 4,
        "One_hot_kernel_size": 4,
        "AtomComp_out_channels_start": 512,
        "DiatomComp_out_channels_start": 116,
        "Global_units": 20,
        "OneHot_out_channels": 1,
        "Concat_units": 94,
        "AtomComp_MaxPool_kernel_size": 2,
        "DiatomComp_MaxPool_kernel_size": 2,
        "OneHot_MaxPool_kernel_size": 10,
        "LRelu_negative_slope": 0.013545684190756122,
        "LRelu_saturation": 40,
        "L1_alpha": 0.000006056805819927765,
        'epochs': 200,
        'delta': 0,
        'device': '0',
        'Use_best_model' : False,
        }

    wandb_run = wandb.init(
        name=config["name"] + "-" + config["time"],
        project="Multi-output_DeepLCCS",
        save_code=False,
        config=config,
    )

    config = wandb.config
    # Get data
    ccs_df_train = pickle.load(open('/home/robbe/multioutput_prediction/data/ccs_df_train.pkl', 'rb'))
    Train_AtomEnc = pickle.load(open("/home/robbe/multioutput_prediction/data/X_train_AtomEnc-multi.pickle", "rb"))
    Train_Globals = pickle.load(
        open("/home/robbe/multioutput_prediction/data/X_train_GlobalFeatures-multi.pickle", "rb")
    )
    Train_DiAminoAtomEnc = pickle.load(open('/home/robbe/multioutput_prediction/data/X_train_DiAminoAtomEnc-multi.pickle', 'rb'))
    Train_OneHot = pickle.load(open('/home/robbe/multioutput_prediction/data/X_train_OneHot-multi.pickle', 'rb'))
    y_train = pickle.load(open('/home/robbe/multioutput_prediction/data/y_train-multi.pickle', 'rb'))

    y_train = np.vstack(y_train)


    # Valid
    ccs_df_valid = pickle.load(open('/home/robbe/multioutput_prediction/data/ccs_df_valid.pkl', 'rb'))
    Valid_AtomEnc = pickle.load(open("/home/robbe/multioutput_prediction/data/X_valid_AtomEnc-multi.pickle", "rb"))
    Valid_Globals = pickle.load(open("/home/robbe/multioutput_prediction/data/X_valid_GlobalFeatures-multi.pickle", "rb"))
    Valid_DiAminoAtomEnc = pickle.load(open('/home/robbe/multioutput_prediction/data/X_valid_DiAminoAtomEnc-multi.pickle', 'rb'))
    Valid_OneHot = pickle.load(open('/home/robbe/multioutput_prediction/data/X_valid_OneHot-multi.pickle', 'rb'))
    y_valid = pickle.load(open('/home/robbe/multioutput_prediction/data/y_valid-multi.pickle', 'rb'))

    y_valid = np.vstack(y_valid)

    # Test
    ccs_df_test = pickle.load(open('/home/robbe/multioutput_prediction/data/ccs_df_test.pkl', 'rb'))
    Test_AtomEnc = pickle.load(open("/home/robbe/multioutput_prediction/data/X_test_AtomEnc-multi.pickle", "rb"))
    Test_Globals = pickle.load(
        open("/home/robbe/multioutput_prediction/data/X_test_GlobalFeatures-multi.pickle", "rb")
    )
    Test_DiAminoAtomEnc = pickle.load(open('/home/robbe/multioutput_prediction/data/X_test_DiAminoAtomEnc-multi.pickle', 'rb'))
    Test_OneHot = pickle.load(open('/home/robbe/multioutput_prediction/data/X_test_OneHot-multi.pickle', 'rb'))
    y_test = pickle.load(open('/home/robbe/multioutput_prediction/data/y_test-multi.pickle', 'rb'))

    y_test = np.vstack(y_test)

    # Set-up GPU device
    device = "cuda:{}".format(config['device']) if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # call model
    model = DeepLC_mimic(config)
    print(model)
    model.to(device)

    wandb.log({"Total parameters": sum(p.numel() for p in model.parameters())})
    # criterion = nn.MSELoss()
    # criterion = nn.SumLoss()
    # criterion = nn.DeltaLoss()
    criterion = LossWithSampleWeights()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Convert the data to PyTorch tensors
    Train_AtomEnc = torch.tensor(Train_AtomEnc, dtype=torch.float32).to(device)
    Train_Globals = torch.tensor(Train_Globals, dtype=torch.float32).to(device)
    Train_DiAminoAtomEnc = torch.tensor(Train_DiAminoAtomEnc, dtype=torch.float32).to(device)
    Train_OneHot = torch.tensor(Train_OneHot, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train).to(device)

    Valid_AtomEnc = torch.tensor(Valid_AtomEnc, dtype=torch.float32).to(device)
    Valid_Globals = torch.tensor(Valid_Globals, dtype=torch.float32).to(device)
    Valid_DiAminoAtomEnc = torch.tensor(Valid_DiAminoAtomEnc, dtype=torch.float32).to(device)
    Valid_OneHot = torch.tensor(Valid_OneHot, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid).to(device)

    Test_AtomEnc = torch.tensor(Test_AtomEnc, dtype=torch.float32).to(device)
    Test_Globals = torch.tensor(Test_Globals, dtype=torch.float32).to(device)
    Test_DiAminoAtomEnc = torch.tensor(Test_DiAminoAtomEnc, dtype=torch.float32).to(device)
    Test_OneHot = torch.tensor(Test_OneHot, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(Train_AtomEnc, Train_DiAminoAtomEnc, Train_Globals, Train_OneHot, y_train)
    valid_dataset = TensorDataset(Valid_AtomEnc, Valid_DiAminoAtomEnc, Valid_Globals, Valid_OneHot, y_valid)
    test_dataset = TensorDataset(Test_AtomEnc, Test_DiAminoAtomEnc, Test_Globals, Test_OneHot, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Train the model
    model = train_model(model, criterion, optimizer, train_loader, valid_loader, config, num_epochs=config['epochs'], device=device)
    # mae, r, perc_95, test_df = evaluate_model(model, test_loader)

    if config['Use_best_model']:
        print('Using best model for evaluation')
        best_model_state_dict = torch.load('models/{}_{}_best_model.pth'.format(config['name'], config['time']))['model_state_dict']
        best_model = DeepLC_mimic(config).to(device)
        best_model.load_state_dict(best_model_state_dict)
        # Evaluation
        mae_best, r_best, perc_95_best, test_df_best = evaluate_model(best_model, test_loader, config, ccs_df_test, info='best')
        plot_predictions(test_df_best, config, mae_best, r_best, perc_95_best, info='best')
    else:
        print('Using last model for evaluation')
        # Evaluation
        mae_best, r_best, perc_95_best, test_df_best = evaluate_model(model, test_loader, config, ccs_df_test, info='last')
        plot_predictions(test_df_best, config, mae_best, r_best, perc_95_best, info='last')

if __name__ == "__main__":
    main()


