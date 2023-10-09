import torch


def cast_all(dtype, *args):
    casted = [arg.to(dtype) for arg in args]
    return casted

class Attention(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, qp, kp, vp, sm_scale=None, causal=True, chunk_size=None):
        orig_dtype = qp.dtype
        q, k, v = cast_all(torch.float32, qp, kp, vp)
        if sm_scale is None:
            sm_scale = q.shape[-1] ** -0.5
        B, H, L, D = q.shape
        if chunk_size is None:
            chunk_size = L
        outputs = torch.zeros_like(q)
        i_coords = torch.linspace(0, L - 1, L).unsqueeze(-1).to(q)
        j_coords = torch.linspace(0, L - 1, L).unsqueeze(0).to(q)
        sums = torch.zeros(B, H, L).to(q)
        for i in range(0, L, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size, :]
            scores = (q_chunk @ k.transpose(-1, -2)) * sm_scale
            if causal:
                mask = (i_coords[i:i+chunk_size] >= j_coords).unsqueeze(0).unsqueeze(0)
                scores = torch.where(mask, scores, 0)
            sums[:, :, i:i+chunk_size] += scores.sum(-1)
            outputs[:, :, i:i+chunk_size] += scores @ v
        outputs /= sums[..., None]
        outputs = outputs.to(orig_dtype)
        # sums doesn't need to be cast, it's an intermediate result
        ctx.save_for_backward(qp, kp, vp, outputs, sums)
        ctx.causal = causal
        ctx.chunk_size = chunk_size
        return outputs

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        qp, kp, vp, output, sums = ctx.saved_tensors
        orig_dtype = qp.dtype
        # sums doesn't need to be cast, it's stored as torch.float32
        q, k, v, output, do = cast_all(torch.float32, qp, kp, vp, output, do)
        causal = ctx.causal
        chunk_size = ctx.chunk_size
        B, H, L, D = q.shape
        if chunk_size is None:
            chunk_size = L
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k) 
        dV = torch.zeros_like(v)
        i_coords = torch.linspace(0, L - 1, L).unsqueeze(-1).to(q)
        j_coords = torch.linspace(0, L - 1, L).unsqueeze(0).to(q)
        delta = (do * output).sum(-1)
        for i in range(0, L, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size, :]
            delta_chunk = delta[:, :, i:i+chunk_size]
            do_chunk = do[:, :, i:i+chunk_size, :]
            s = (q_chunk @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
            s /= sums[:, :, i:i+chunk_size, None]
            if causal:
                mask = (i_coords[i:i+chunk_size] >= j_coords).unsqueeze(0).unsqueeze(0)
                s = torch.where(mask, s, 0)
            dP = do_chunk @ v.transpose(-1, -2)
            dS = (dP - delta_chunk[..., None]) / sums[:, :, i:i+chunk_size, None] * q.shape[-1] ** -0.5
            if causal:
                dS = torch.where(mask, dS, 0)
            dV += s.transpose(-1, -2) @ do_chunk
            dQ[:, :, i:i+chunk_size, :] += dS @ k
            dK += (q_chunk.transpose(-1, -2) @ dS).transpose(-1, -2)
        dQ, dK, dV = cast_all(orig_dtype, dQ, dK, dV)
        return dQ, dK, dV, None, None, None

low_mem_attn = Attention.apply
