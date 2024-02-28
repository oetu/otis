from typing import Any, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import util.transformations as transformations
import util.augmentations as augmentations


class SignalDataset(Dataset):
    """
    Unimodal dataset that generates views of signals.
    """
    def __init__(self, data_path, labels_path=None, labels_mask_path=None, downstream_task:str=None, 
                 finetune=False, train=False, modality_offsets:Dict=None, args=None) -> None:
        """
            labels_path: path to labels (finetuning / online evaluation)
            labels_mask_path: path to labels masks (finetuning / online evaluation)
            downstream_task: downstream task (finetuning / online evaluation)

            modality_offsets: offsets for positional embedding Y
        """
        data = torch.load(data_path, map_location=torch.device('cpu')) # load to ram

        self.finetune = finetune
        if finetune:
            # finetuning / online evaluation
            modality = [("ecg", sample.unsqueeze(0)[..., :args.input_electrodes, :].shape) for sample in data]
            data = [sample.unsqueeze(0)[..., :args.input_electrodes, :] for sample in data]
        else:
            # pretraining
            modality = [(sample[0], sample[1].shape) for sample in data]
            data = [sample[1] for sample in data]

        self.modality = modality
        self.modalities = {modality: shape for modality, shape in sorted(list(set(self.modality)))} # unique modalities

        modality_list = [mod[0] for mod in modality]
        unique_modalities = list(set(modality_list))
        
        self.modality_weights = {}
        for mod_current in unique_modalities:
            mod_indices = torch.tensor([mod == mod_current for mod in modality_list])
            mod_weight = len(modality) / (len(unique_modalities) * mod_indices.sum())
            self.modality_weights.update( {mod_current: mod_weight} )

        self.offsets = {}
        if modality_offsets is None:
            offset = 0
            for modality, shape in self.modalities.items():
                self.offsets.update( {modality: offset} )
                offset += shape[-2]
        else:
            self.offsets = modality_offsets

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

    def set_modality_offsets(self, modality_offsets:Dict=None):
        """set predefined modality offsets"""
        self.offsets = modality_offsets

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        data = self.data[idx]        
        if self.train == False:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
            ])
            # transform = torch.nn.Identity()
        else:
            transform = transforms.Compose([
                augmentations.CropResizing(fixed_crop_len=self.args.time_steps, resize=False),
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

        modality, _ = self.modality[idx]
            
        return data, label, label_mask, self.args.patch_size, self.offsets[modality], modality

    @staticmethod
    def collate_fn(batch):
        # determine the biggest size in the batch
        shape = [sample[0].shape for sample in batch]
        max_values = [max(x) for x in zip(*shape)]
        max_channels = max_values[1] 
        max_timesteps = max_values[2]

        # Zero pad the input data to the biggest size 
        # (B, 1, C_max, T_max)
        data = [torch.nn.functional.pad(sample[0], 
                                        pad=(0, int(max_timesteps - sample[0].shape[-1]), 0, int(max_channels - sample[0].shape[-2])), 
                                        mode="constant", value=0) for sample in batch]
        data = torch.stack(data, dim=0)

        # (p, q)
        patch_size = batch[0][3]

        grid_width = torch.tensor([sample[0].shape[-1] // patch_size[-1] for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

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

        modality = [sample[5] for sample in batch]
    
        return data, patch_size, attn_mask, torch.LongTensor(pos_embed_y), modality
    
    @staticmethod
    def collate_fn_ft(batch):
        data = torch.stack([sample[0] for sample in batch], dim=0)
        label = torch.stack([sample[1] for sample in batch], dim=0)
        label_mask = torch.stack([sample[2] for sample in batch], dim=0)

        grid_width = batch[0][0].shape[-1] // batch[0][3][-1]
        grid_height = batch[0][0].shape[-2] // batch[0][3][-2]
        pos_embed_y = torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width) + 1 + batch[0][4]

        return data, label, label_mask, torch.LongTensor(pos_embed_y)