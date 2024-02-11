import torch


def norm(data:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalize data to have mean=0 and standard_deviation=1

    Parameters
    ----------
    data:  tensor
    """
    mean = torch.mean(data, dim=-1, keepdim=True)
    var = torch.var(data, dim=-1, keepdim=True)

    return (data - mean) / (var + 1e-9)**0.5

def ncc(data_0:torch.Tensor(), data_1:torch.Tensor()) -> torch.Tensor():
    """
    Zero-Normalized cross-correlation coefficient between two data sets

    Zero-Normalized cross-correlation equals the cosine of the angle between the unit vectors F and T, 
    being thus 1 if and only if F equals T multiplied by a positive scalar. 

    Parameters
    ----------
    data_0, data_1 :  tensors of same size
    """

    nb_of_signals = 1
    for dim in range(data_0.dim() - 1): # all but the last dimension (which is the actual signal)
        nb_of_signals = nb_of_signals * data_0.shape[dim]

    cross_corrs = (1.0 / (data_0.shape[-1] - 1)) * torch.sum(norm(data=data_0) * norm(data=data_1), dim=-1)

    return (cross_corrs.sum() / nb_of_signals)

def masked_mean(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the mean while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(tensor_masked) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) + 1e-9)
        else:
            numerator = torch.sum(tensor_masked, dim=dim, keepdim=keep_dim)
            denominator = torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) + 1e-9
            return numerator / denominator

def masked_var(tensor, attn_mask=None, dim=None, keep_dim=False):
    """
        Determine the variance while considering the attention mask

        tensor: (B, N, D)
        attn_mask: (B, N)
    """
    if attn_mask is None:
        if dim is None:
            return torch.var(tensor)
        else:
            return torch.var(tensor, dim=dim, keepdim=keep_dim)
    else:
        tensor_masked = tensor * attn_mask.unsqueeze(-1)
        deviations = tensor_masked - masked_mean(tensor, attn_mask, dim=dim, keep_dim=True)
        deviations = deviations * attn_mask.unsqueeze(-1)
        squared_deviations = (deviations ** 2) * attn_mask.unsqueeze(-1)
        if dim is None:
            return torch.sum(squared_deviations) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1])) - 1 + 1e-9)
        else:
            return torch.sum(squared_deviations, dim=dim, keepdim=keep_dim) / (torch.sum(attn_mask.unsqueeze(-1).repeat(1, 1, tensor.shape[-1]), dim=dim, keepdim=keep_dim) - 1 + 1e-9)