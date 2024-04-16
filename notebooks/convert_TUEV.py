import os

import torch
import numpy as np
import pickle

from multiprocessing import Pool
import time


def list_files(rootdir, file_format=".edf"):
    files = []
    for file in os.listdir(rootdir):
        curr_object = os.path.join(rootdir, file)
        if os.path.isdir(curr_object): 
            files += list_files(curr_object)
        elif file_format in curr_object:
            files.append(curr_object)

    return files


def get_data_and_label(file_path):
    file = open(file_path, "rb")
    pkl_object = pickle.load(file)

    data = pkl_object["signal"]
    label = pkl_object["label"].item() - 1

    return data, label


def main():
    data_path_train = '/home/oturgut/data/TUEV/v2.0.1/edf/processed/processed_train'
    data_path_val = '/home/oturgut/data/TUEV/v2.0.1/edf/processed/processed_eval'
    data_path_test = '/home/oturgut/data/TUEV/v2.0.1/edf/processed/processed_test'

    files_train = list_files(data_path_train, ".pkl")
    # files_val = list_files(data_path_val, ".pkl")
    # files_test = list_files(data_path_test, ".pkl")

    start = time.time()

    num_processes = 24
    with Pool(processes=num_processes) as pool:
        results = pool.map(get_data_and_label, files_train)

    total = time.time() - start
    print(f"Pre-processing time: {total}")

    # Extract results into separate lists
    data_raw, labels_raw = zip(*results)

    data = [("eeg_10-20", torch.tensor(sample, dtype=torch.float32)) for sample in data_raw]
    labels = torch.nn.functional.one_hot(torch.tensor(labels_raw, dtype=torch.int64), num_classes=6).to(torch.int32)

    print(f"Number of samples: {len(data)}")
    print(f"Data shape: {data[0][1].shape}")
    print(f"Labels shape: {labels.shape}")

    torch.save(data, "/home/oturgut/data/processed/TUEV/train/data.pt")
    torch.save(labels, "/home/oturgut/data/processed/TUEV/train/labels.pt")


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    main()