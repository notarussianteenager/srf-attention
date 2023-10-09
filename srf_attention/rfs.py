import math
import torch
from scipy.linalg import hadamard

# generate a hadamard matrix of size d x d
@torch.no_grad()
def orthogonal_matrix_chunk_hadamard(d, device = None):
    (d1,d2,d3)=(torch.diag(torch.sign(torch.randn(d, dtype=torch.float32)-0.5)),
                torch.diag(torch.sign(torch.randn(d, dtype=torch.float32)-0.5)),
                torch.diag(torch.sign(torch.randn(d, dtype=torch.float32)-0.5)))
    (h1,h2,h3) =[torch.tensor(hadamard(d)).to(d1) for _ in range(3)]
    m = torch.matmul(torch.matmul(h3,d3),torch.matmul(torch.matmul(h2,d2),torch.matmul(h1,d1)))
    m = m / torch.sqrt(torch.sum(m[0]**2))
    m = m.to(device)
    return m

# compute dirs
@torch.no_grad()
def compute_simplex_dir(d, device = None):
    simp_dir = torch.diag(torch.ones(d, dtype=torch.float32))/math.sqrt(2) - torch.ones((d,d), dtype=torch.float32)*(1/((d-1) * math.sqrt(2))) *(1 + 1/math.sqrt(d)) #begin by getting the proj directions
    simp_dir[d-1,:] = 1/math.sqrt(2 * d) * torch.ones((d), dtype=torch.float32)
    simp_dir[:,d-1] = 0
    simp_dir = simp_dir / math.sqrt(torch.sum(simp_dir[1,:]**2))
    chi2 = torch.distributions.chi2.Chi2(torch.ones(d, dtype=torch.float32))
    rand_sim = torch.matmul(torch.diag(torch.sqrt(chi2.sample().to(simp_dir))),simp_dir)
    rand_sim = rand_sim.to(device)
    return rand_sim

# generate rfs
@torch.no_grad()
def simplex_random_matrix(nb_rows, nb_columns, scale = False, normalize = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk_hadamard(nb_columns, device = device)
        q = torch.matmul(compute_simplex_dir(nb_columns, device = device), q)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk_hadamard(nb_columns, device = device)
        q = torch.matmul(compute_simplex_dir(nb_columns, device = device), q)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list).to(device).float()
    if scale:
        final_matrix /= math.sqrt(nb_columns)
    if normalize:
        final_matrix /= final_matrix.norm(p=2, dim=-1)[..., None]
    return final_matrix

