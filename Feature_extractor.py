import pandas as pd
from collections import Counter
import random
import deeplcretrainer.cnn_functions
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extractor for DeepLCCS")
    parser.add_argument("trainset", type=str, help="Path to the training set")
    parser.add_argument("--testset", type=str, default=None, help="Path to the testing set")
    parser.add_argument("--validset", type=str, default=None, help="Path to the validation set")
    parser.add_argument("--info", type=str, default="", help="Information for the output files")
    parser.add_argument("--test_needed", type=bool, default=False, help="Whether or not to split the data in a training and testing set")
    parser.add_argument("--valid_needed", type=bool, default=False, help="Whether or not to split the data in a training and validation set")
    parser.add_argument("--valid_split", type=float, default=0.1, help="Fraction of the data to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of the data to use for testing")
    parser.add_argument('--output_dir', type=str, default='../data_clean/', help='Directory to save the output files')
    return parser.parse_args()

def train_test_split(ccs_df, test_split=0.1, output_dir="./", info=""):
    X_matrix_count = pd.DataFrame(ccs_df["seq"].apply(Counter).to_dict()).fillna(0.0).T
    # Get all the index identifiers
    all_idx = list(X_matrix_count.index)
    random.seed(42)

    # Shuffle the index identifiers so we can randomly split them in a testing and training set
    random.shuffle(all_idx)

    # Select 90 % for training and the remaining 10 % for testing
    train_idx = all_idx[0 : int(len(all_idx) * (1 - test_split))]
    test_idx = all_idx[int(len(all_idx) * (1 - test_split)) :]

    # Get the train and test indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_test = ccs_df.loc[test_idx, :]
    # ccs_df_train.to_pickle("{}/ccs_df_train_{}.pkl".format(output_dir, info))
    # ccs_df_test.to_pickle("{}/ccs_df_test_{}.pkl".format(output_dir, info))
    return ccs_df_train, ccs_df_test

def one_hot_encode(charge):
    if charge == 1:
        return [1, 0, 0, 0, 0, 0]
    elif charge == 2:
        return [0, 1, 0, 0, 0, 0]
    elif charge == 3:
        return [0, 0, 1, 0, 0, 0]
    elif charge == 4:
        return [0, 0, 0, 1, 0, 0]
    elif charge == 5:
        return [0, 0, 0, 0, 1, 0]
    elif charge == 6:
        return [0, 0, 0, 0, 0, 1]

def get_features(args):
    try:
        ccs_df_train = pd.read_pickle(args.trainset)
    except ValueError:
        ValueError("No training set provided")

    try:
        ccs_df_valid = pd.read_pickle(args.validset)
    except ValueError:
        if args.valid_needed:
            ccs_df_train, ccs_df_valid = train_test_split(ccs_df_train, args.valid_split, args.output_dir, args.info)
            ccs_df_valid.to_pickle("{}/ccs_df_valid_{}.pkl".format(args.output_dir, args.info))
            ccs_df_train.to_pickle("{}/ccs_df_train_{}.pkl".format(args.output_dir, args.info))
        else:
            ccs_df_valid = None

    try:
        ccs_df_test = pd.read_pickle(args.testset)
    except:
        if args.test_needed:
            ccs_df_train, ccs_df_test = train_test_split(ccs_df_train, args.test_split, args.output_dir)
            ccs_df_test.to_pickle("{}/ccs_df_test_{}.pkl".format(args.output_dir, args.info))
        else:
            ccs_df_test = None

    train_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_train, predict_ccs=True)
    train_df["charge"] = ccs_df_train["charge"]
    train_df["seq"] = ccs_df_train["seq"]
    train_df["modifications"] = ccs_df_train["modifications"]
    (
        X_train,
        X_train_sum,
        X_train_global,
        X_train_hc,
        y_train_FAKE,
    ) = deeplcretrainer.cnn_functions.get_feat_matrix(train_df)

    y_train = np.array(ccs_df_train["CCS"])

    # X_train_global = np.concatenate((X_train_global, np.array([one_hot_encode(x) for x in train_df["charge"]])), axis=1)
    # X_train_global = np.concatenate((X_train_global, np.array([[1] if "0|Acetyl" in x else [0] for x in train_df["modifications"]])), axis=1)

    train_data = {'X_train_AtomEnc' : X_train, 'X_train_DiAminoAtomEnc' : X_train_sum, 'X_train_GlobalFeatures' : X_train_global, 'X_train_OneHot' : X_train_hc, 'y_train' : y_train}

    if ccs_df_valid is not None:
        valid_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_valid, predict_ccs=True)
        valid_df["charge"] = ccs_df_valid["charge"]
        valid_df["seq"] = ccs_df_valid["seq"]
        valid_df["modifications"] = ccs_df_valid["modifications"]

        (
            X_valid,
            X_valid_sum,
            X_valid_global,
            X_valid_hc,
            y_valid_FAKE,
        ) = deeplcretrainer.cnn_functions.get_feat_matrix(valid_df)

        # X_valid_global = np.concatenate((X_valid_global, np.array([one_hot_encode(x) for x in valid_df["charge"]])), axis=1)
        # X_valid_global = np.concatenate((X_valid_global, np.array([[1] if "0|Acetyl" in x else [0] for x in valid_df["modifications"]])), axis=1)

        y_valid = np.array(ccs_df_valid["CCS"])

        valid_data = {'X_valid_AtomEnc' : X_valid, 'X_valid_DiAminoAtomEnc' : X_valid_sum, 'X_valid_GlobalFeatures' : X_valid_global, 'X_valid_OneHot' : X_valid_hc, 'y_valid' : y_valid}

    if ccs_df_test is not None:
        test_df = deeplcretrainer.cnn_functions.get_feat_df(ccs_df_test, predict_ccs=True)
        test_df["charge"] = ccs_df_test["charge"]
        test_df["seq"] = ccs_df_test["seq"]
        test_df["modifications"] = ccs_df_test["modifications"]

        (
        X_test,
        X_test_sum,
        X_test_global,
        X_test_hc,
        y_test_FAKE,
        ) = deeplcretrainer.cnn_functions.get_feat_matrix(test_df)

        # X_test_global = np.concatenate((X_test_global, np.array([one_hot_encode(x) for x in test_df["charge"]])), axis=1)
        # X_test_global = np.concatenate((X_test_global, np.array([[1] if "0|Acetyl" in x else [0] for x in test_df["modifications"]])), axis=1)

        y_test = np.array(ccs_df_test["CCS"])

        test_data = {'X_test_AtomEnc' : X_test, 'X_test_DiAminoAtomEnc' : X_test_sum, 'X_test_GlobalFeatures' : X_test_global, 'X_test_OneHot' : X_test_hc, 'y_test' : y_test}

    if (ccs_df_valid is not None) and (ccs_df_test is not None):
        return train_data, valid_data, test_data
    elif ccs_df_valid is not None:
        return train_data, valid_data
    elif ccs_df_test is not None:
        return train_data, test_data
    else:
        return train_data,

def main():
    args = parse_args()
    datasets = get_features(args)
    output_dir = args.output_dir

    for dataset in datasets:
        for key in dataset:
            pickle.dump(dataset[key], open("{}/{}-{}.pickle".format(output_dir, key, args.info), "wb"))

if __name__ == "__main__":
    main()
