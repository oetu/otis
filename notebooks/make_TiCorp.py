import time

import torch
import numpy as np

import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy


# Load data
print("LOAD DATA\n")
filename = "/vol/aimspace/projects/physionet/mimic/processed/mimic-ecg-text/ecgs_memmap.dat"
ecg_data = np.memmap(filename, dtype='float32', mode='r+', shape=(800035, 12, 5000))

print("Load remaining data\n")
other_train_path = f"/vol/aimspace/users/tuo/data/processed/TiCorp/train_wo_mimic_new.pt"
other_train = torch.load(other_train_path)

other_val_path = f"/vol/aimspace/users/tuo/data/processed/TiCorp/val_wo_mimic_new.pt"
other_val = torch.load(other_val_path)


def create_ticorp(idx):
    """Create TiCorp data."""
    return deepcopy(ecg_data[idx])


if __name__ == '__main__':
    start_time = time.time()

    # Define the number of processes
    num_processes = 32

    lower_bnd = 0
    upper_bnd = 400000 # 800035

    # Create a multiprocessing Pool
    print("PROCESS DATA")
    with multiprocessing.Pool(processes=num_processes) as pool:
        ecgs_mimic = pool.map(create_ticorp, [idx for idx in range(lower_bnd, upper_bnd)])

    print(f"Number of processed data points: {len(ecgs_mimic)}")

    ecgs_mimic_train = [("ecg", torch.from_numpy(sample)) for sample in ecgs_mimic[:int(0.975*upper_bnd)]]
    # ecgs_mimic_train = [("ecg", torch.from_numpy(sample)) for sample in ecgs_mimic]
    ticorp_train = ecgs_mimic_train + other_train

    ecgs_mimic_val = [("ecg", torch.from_numpy(sample)) for sample in ecgs_mimic[int(0.975*upper_bnd):]]
    # ecgs_mimic_val = [("ecg", torch.from_numpy(sample)) for sample in ecgs_mimic]
    ticorp_val = ecgs_mimic_val + other_val

    # Save data
    output_path = f"/vol/aimspace/users/tuo/data/processed/TiCorp"
    print(f"SAVE DATA ({output_path})")
    # torch.save(ticorp_train, f"{output_path}/train_all_new.pt", pickle_protocol=4)
    # torch.save(ticorp_val, f"{output_path}/val_all_new.pt", pickle_protocol=4)

    total_time = time.time() - start_time
    print(f"Processing time: {total_time}\n")