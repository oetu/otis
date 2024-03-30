import time

import torch
import numpy as np
import pandas as pd

import wfdb
from scipy import sparse
from scipy.sparse.linalg import spsolve

import multiprocessing


class Normalization(object):
    """
    Normalize the data.
    """
    def __init__(self, mode="sample_wise", groups=[3, 6, 12]) -> None:
        self.mode = mode
        self.groups = groups

    def __call__(self, sample) -> np.array:
        sample_dtype = sample.dtype

        if self.mode == "sample_wise":
            mean = np.mean(sample)
            var = np.var(sample)
        
        elif self.mode == "channel_wise":
            mean = np.mean(sample, axis=-1, keepdims=True)
            var = np.var(sample, axis=-1, keepdims=True)
        
        elif self.mode == "group_wise":
            mean = []
            var = []

            lower_bound = 0
            for idx in self.groups:
                mean_group = np.mean(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                mean_group = np.repeat(mean_group, repeats=int(idx-lower_bound), axis=0)
                var_group = np.var(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                var_group = np.repeat(var_group, repeats=int(idx-lower_bound), axis=0)
                lower_bound = idx

                mean.extend(mean_group)
                var.extend(var_group)

            mean = np.array(mean, dtype=sample_dtype)
            var = np.array(var, dtype=sample_dtype)

        normalized_sample = (sample - mean) / (var + 1.e-5)**.5

        return normalized_sample
    

def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Paul H. C. Eilers and Hans F.M. Boelens: Baseline Correction with Asymmetric Least Squares Smoothing
    https://www.rdocumentation.org/packages/baseline/versions/1.3-1/topics/baseline.als
    """
    L = len(y)
    D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


# define data path
ecg_path = '/home/oturgut/data/mimic-iv-ecg/1.0/'
ecg_detail = pd.read_csv(ecg_path+'record_list.csv')


def process_ecgs(sample_id):
    file_name = ecg_detail.loc[sample_id, "path"]
    rec_path = ecg_path+file_name

    data = wfdb.rdsamp(rec_path, return_res=32)[0]
    data = np.transpose(data)

    # remove nan
    data = np.nan_to_num(data)
    
    # clamp
    data_std = data.std()
    data = np.clip(data, a_min=-4*data_std, a_max=4*data_std)

    # remove baseline wander
    baselines = np.zeros_like(data)
    for lead in range(data.shape[0]):
        baselines[lead] = baseline_als(data[lead], lam=1e7, p=0.3, niter=5)
    data = data - baselines

    # normalize data
    transform = Normalization(mode="group_wise", groups=[3, 6, 12])
    data = transform(data)

    return data, rec_path


if __name__ == '__main__':
    start_time = time.time()

    # Create numpy memory-map file
    filename = "/home/oturgut/data/processed/mimic-ecg-text/ecgs_memmap.dat"
    ecg_data = np.memmap(filename, dtype='float32', mode='r+', shape=(800035, 12, 5000))

    # Define the number of processes
    num_processes = 32

    # Create a multiprocessing Pool
    lower_bnd = 725000
    upper_bnd = 800035
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_ecgs, range(lower_bnd, upper_bnd))
        
    # extract results into separate lists
    ecgs, rec_paths = zip(*results)

    print(len(ecgs))

    # tensor_ecg = torch.tensor(np.array(ecgs))
    # print(tensor_ecg.shape, tensor_ecg.dtype)

    # Store numpy arrays into memory-map file
    ecg_data[lower_bnd:upper_bnd] = np.array(ecgs)
    ecg_data.flush()

    # torch.save(tensor_ecg, "/home/oturgut/data/processed/mimic-ecg-text/ecgs.pt")
    # torch.save(rec_paths, "/home/oturgut/data/processed/mimic-ecg-text/ecgs_file_path.pt")

    total_time = time.time() - start_time
    print(f"Processing time: {total_time}")