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


def create_ticorp(idx):
    """Create TiCorp data."""
    return deepcopy(ecg_data[idx])


if __name__ == '__main__':
    start_time = time.time()

    # Define the number of processes
    num_processes = 32

    lower_bnd = 790035
    upper_bnd = 800035
    
    # Create a multiprocessing Pool
    print("PROCESS DATA")
    with multiprocessing.Pool(processes=num_processes) as pool:
        mimic = pool.map(create_ticorp, [idx for idx in range(lower_bnd, upper_bnd)])

    print(f"Number of processed data points: {len(mimic)}")

    mimic = [("ecg", torch.from_numpy(sample)) for sample in mimic]

    # Save data
    output_path = f"/vol/aimspace/users/tuo/data/processed/TiCorp/mimic_{upper_bnd}.pt"
    print(f"SAVE DATA ({output_path})")
    torch.save(mimic, f"{output_path}", pickle_protocol=4)

    total_time = time.time() - start_time
    print(f"Processing time: {total_time}\n")