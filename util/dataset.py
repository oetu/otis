from typing import Any, Tuple, Dict

import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.augmentations as augmentations


class TimeSeriesDataset(Dataset):
    """
    Dataset for multi-domain time series analysis.
    """
    def __init__(self, data_path:str, labels_path:str=None, labels_mask_path:str=None, 
                 downstream_task:str=None, 
                 domain_offsets:Dict=None, domain_agnostic:str=False, 
                 train:bool=False, args=None) -> None:
        """
            labels_path: path to labels (finetuning / online evaluation)
            labels_mask_path: path to labels masks (finetuning / online evaluation)
            downstream_task: downstream task (finetuning / online evaluation)

            domain_offsets: offsets for pos_embed_y
            domain_agnostic: share pos_embed_y across domains
        """
        data = torch.load(data_path, map_location=torch.device('cpu')) # load to ram

        # .unsqueeze(0) to add auxiliary channel (similar to rgb in imgs)
        domain = [(sample[0], sample[1].unsqueeze(0).shape) for sample in data]
        data = [sample[1].unsqueeze(0) for sample in data]
        # if train:
        #     data = [sample[1][..., :10452].unsqueeze(0) for sample in data[:1]]
        #     # data = [sample[1][..., :676].unsqueeze(0) for sample in data[:1]]
        # if not train:
        #     # data = [sample[1][..., :10452].unsqueeze(0) for sample in data[:1]]
        #     # data = [sample[1][..., 10452:13936].unsqueeze(0) for sample in data[:1]]
        #     data = [sample[1][..., 13936:].unsqueeze(0) for sample in data[:1]]
        #     # data = [sample[1][..., 773:].unsqueeze(0) for sample in data[:1]]

        self.domain_agnostic = domain_agnostic

        self.domain = domain
        self.domains = {domain: shape for domain, shape in sorted(list(set(self.domain)))} # unique domains

        domain_list = [mod[0] for mod in domain]
        unique_domains = list(set(domain_list))
        
        self.domain_weights = {}
        for mod_current in unique_domains:
            mod_indices = torch.tensor([mod == mod_current for mod in domain_list])
            mod_weight = len(domain) / (len(unique_domains) * mod_indices.sum())
            self.domain_weights.update( {mod_current: mod_weight} )

        self.offsets = {}
        if domain_offsets is None:
            offset = 0
            for domain, shape in self.domains.items():
                self.offsets.update( {domain: offset} )
                if not self.domain_agnostic:
                    offset += shape[-2]
        else:
            self.offsets = domain_offsets

        self.data = data

        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu')) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

        self.downstream_task = downstream_task
        self.train = train 
        self.args = args

    def set_domain_offsets(self, domain_offsets:Dict=None):
        """set predefined domain offsets"""
        self.offsets = domain_offsets

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        # (1, C, T)
        data = self.data[idx]
        if self.train == False:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
            ])
        else:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_resize_len=self.args.time_steps, 
                                           lower_bnd=self.args.crop_lower_bnd, 
                                           upper_bnd=self.args.crop_upper_bnd,
                                           resize=True),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
            ])

        data = transform(data)

        if self.downstream_task == 'regression':
            label = self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd]
            label_mask = self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            label = self.labels[idx].type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        domain, _ = self.domain[idx]
        
        return data, label, label_mask, self.args.patch_size, self.offsets[domain], domain, self.args.time_steps

    @staticmethod
    def collate_fn(batch):
        # (p, q)
        patch_size = batch[0][3]
        grid_width = torch.tensor([sample[0].shape[-1] // patch_size[-1] for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

        # Determine the largest shape in the batch
        shape = [sample[0].shape for sample in batch]
        max_values = [max(x) for x in zip(*shape)]
        max_channels = max_values[-2]
        max_timesteps = min(((max_values[-1] // patch_size[-1]) + 1) * patch_size[-1], batch[0][6]) # multiple of q 

        if grid_width.max() * patch_size[-1] < batch[0][6]:
            grid_width = grid_width + 1

        # Zero pad the input data to the largest shape 
        # (B, 1, C_max, T_max)
        data = [torch.nn.functional.pad(sample[0], 
                                        pad=(0, int(max_timesteps - sample[0].shape[-1]), 0, int(max_channels - sample[0].shape[-2])), 
                                        mode="constant", value=0) for sample in batch]
        data = torch.stack(data, dim=0)

        # Create the attention mask 
        # (B, C'_max, T'_max), with C'_max=C_max/p, T'_max=T_max/p
        attn_mask = [torch.nn.functional.pad(torch.ones(size=(grid_height[idx], grid_width[idx])), 
                                             pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])), 
                                             mode="constant", value=0) for idx in range(len(batch))]
        attn_mask = torch.stack(attn_mask, dim=0)
        
        # Create the pos embedding Y
        # (B, C'_max, T'_max)
        pos_embed_y = [torch.nn.functional.pad(torch.arange(grid_height[idx]).view(-1, 1).repeat(1, grid_width[idx]) + 1 + sample[4],
                                               pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])), 
                                               mode="constant", value=0) for idx, sample in enumerate(batch)]
        pos_embed_y = torch.stack(pos_embed_y, dim=0)

        domain = [sample[5] for sample in batch]
    
        return data, attn_mask, torch.LongTensor(pos_embed_y), domain
    
    @staticmethod
    def collate_fn_ft(batch):
        data = torch.stack([sample[0] for sample in batch], dim=0)
        label = torch.stack([sample[1] for sample in batch], dim=0)
        label_mask = torch.stack([sample[2] for sample in batch], dim=0)

        grid_width = batch[0][0].shape[-1] // batch[0][3][-1]
        grid_height = batch[0][0].shape[-2] // batch[0][3][-2]
        pos_embed_y = torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width) + 1 + batch[0][4]

        return data, label, label_mask, torch.LongTensor(pos_embed_y)