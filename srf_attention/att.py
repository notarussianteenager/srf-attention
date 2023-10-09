import torch
from .rfs import simplex_random_matrix
from .kernel import softmax_kernel_bhld, softmax_kernel_blhd
from .torch_low_mem_attn import low_mem_attn
from functools import partial
try:
    import flash_attn_no_softmax
except:
    pass

class Attention(torch.nn.Module):
    def __init__(self, d, causal, n_features = None, scale = False, normalize = False, device = None):
        super().__init__()
        self.causal = causal
        self.n_features = d if n_features is None else n_features
        self.d = d
        self.scale = scale
        self.normalize = normalize
        self.redraw_on_call = False
        self.draw_features = partial(simplex_random_matrix, nb_rows = self.d, nb_columns = self.n_features, scale = self.scale, normalize = self.normalize)
        projection_matrix = self.draw_features(device = device)
        self.register_buffer('projection_matrix', projection_matrix)

    def redraw_on_call_(self, value=True):
        self.redraw_on_call = value

    def redraw_(self):
        projection = self.draw_features(device=self.projection_matrix.device)
        self.projection_matrix.copy_(projection)
        del projection

    # (B, H, L, D)
    def forward_train_torch(self, q, k, v, chunk_size=None):
        q = softmax_kernel_bhld(data=q, projection_matrix=self.projection_matrix, is_query=True, normalize_data=True, eps=1e-4)
        k, key_maxima = softmax_kernel_bhld(data=k, projection_matrix=self.projection_matrix, is_query=False, normalize_data=True, eps=1e-4)
        o = low_mem_attn(q, k, v, None, self.causal, chunk_size)
        return o

    # (B, L, H, D)
    def forward_train_flash(self, q, k, v):
        q = softmax_kernel_blhd(data=q, projection_matrix=self.projection_matrix, is_query=True, normalize_data=True, eps=1e-4)
        k, key_maxima = softmax_kernel_blhd(data=k, projection_matrix=self.projection_matrix, is_query=False, normalize_data=True, eps=1e-4)
        o = flash_attn_no_softmax.flash_attn_func(q, k, v, softmax_scale=1.0, causal=self.causal)
        return o

    # q, k, v should be of shape (b, h, l, d)
    def forward(self, q=None, k=None, v=None, mode = None, attn_fn = None, chunk_size=None):
        assert mode in ['train'] and attn_fn in ['torch', 'flash']
        if self.redraw_on_call:
            self.redraw_(device = q.device)
        
        if mode == 'train':
            if attn_fn == 'torch':
                o = self.forward_train_torch(q, k, v, chunk_size=chunk_size)
                return o
            elif attn_fn == 'flash':
                print('Warning: using Flash Attention for SRF fine-tuning may cause arithmetic failure.')
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                o = self.forward_train_flash(q, k, v)
                return o.transpose(1, 2)
