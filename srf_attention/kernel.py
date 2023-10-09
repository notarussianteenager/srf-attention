import torch
from einops import repeat

def softmax_kernel_bhld(data, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    b, h, l, d = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data.pow(2).sum(-1).mul(0.5 * data_normalizer ** 2).unsqueeze(-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
        return data_dash.type_as(data)
    else:
        key_maxima = torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - key_maxima) + eps)
        return data_dash.type_as(data), key_maxima.type_as(data)

def softmax_kernel_blhd(data, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    b, l, h, d = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = projection_matrix.shape[0] ** -0.5

    projection = repeat(projection_matrix, 'j d -> b l j d', b = b, l = l)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data.pow(2).sum(-1).mul(0.5 * data_normalizer ** 2).unsqueeze(-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
        return data_dash.type_as(data)
    else:
        key_maxima = torch.amax(data_dash, dim=(-1, -3), keepdim=True).detach()
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - key_maxima) + eps)
        return data_dash.type_as(data), key_maxima.type_as(data)

