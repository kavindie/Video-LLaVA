from dppy.finite_dpps import FiniteDPP
import torch
import numpy as np

from torch import log, solve

import numpy as np

def greedy_map_dpp(L, k=None):
    """
    Greedy MAP (Maximum A Posteriori) inference for DPP.
    
    Args:
        L: L matrix of shape [m, m]
        k: number of items to select (optional)
    
    Returns:
        selected_items: list of indices of selected items
    """
    n = L.shape[0]
    
    ground_set = list(range(n))
    selected_items = []

    while k is None or len(selected_items) < k:
        if not ground_set:
            break

        best_item = None
        best_det_ratio = -np.inf  # allow any improvement

        for item in ground_set:
            new_selected = selected_items + [item]

            if len(selected_items) == 0:
                det_ratio = L[item, item]
            else:
                L_Y = L[np.ix_(selected_items, selected_items)]
                L_Y_x = L[np.ix_(selected_items, [item])]
                L_x_Y = L_Y_x.T
                L_xx = L[item, item]
                
                try:
                    schur = L_xx - L_x_Y @ np.linalg.solve(L_Y, L_Y_x)
                    det_ratio = schur
                except np.linalg.LinAlgError:
                    det_ratio = -np.inf  # Skip if L_Y is singular

            if det_ratio > best_det_ratio:
                best_det_ratio = det_ratio
                best_item = item

        if best_item is None or best_det_ratio <= 0:
            break

        selected_items.append(best_item)
        ground_set.remove(best_item)

    return selected_items

import numpy as np

def greedy_map_dpp_fast(L, k=None, eps=1e-10):
    """
    Fast greedy MAP inference for DPP using Cholesky updates.
    
    Args:
        L: L-ensemble matrix (must be symmetric PSD)
        k: number of items to select (optional)
        eps: small value for numerical stability
    
    Returns:
        selected_items: list of selected item indices
    """
    n = L.shape[0]
    cis = np.zeros((0, n))  # to store intermediate solutions
    di2s = np.copy(np.diag(L))  # initial marginal gains
    selected_items = []

    while k is None or len(selected_items) < k:
        if len(selected_items) == n or np.all(di2s < eps):
            break

        # Choose the item with max marginal gain
        j = np.argmax(di2s)
        selected_items.append(j)

        # Update cached values
        cj = (L[j] - cis.T @ cis[:, j]) / np.sqrt(di2s[j])
        cis = np.vstack([cis, cj])
        di2s -= cj ** 2
        di2s[j] = -np.inf  # ensure it won't be picked again

    return selected_items


import torch

import torch

def greedy_map_dpp_fast_torch(L, k=None, eps=1e-10):
    """
    Fast greedy MAP inference for DPP using Cholesky updates (PyTorch version).
    
    Args:
        L: L-ensemble matrix (torch tensor) [n x n], symmetric and PSD
        k: number of items to select (optional)
        eps: small threshold for stopping
    
    Returns:
        selected_items: list of selected indices
    """
    n = L.shape[0]
    device = L.device

    cis = torch.zeros((0, n), device=device)  # Cache for Cholesky-like vectors
    di2s = torch.diag(L).clone()              # Initial marginal gains
    selected_items = []

    while k is None or len(selected_items) < k:
        if len(selected_items) == n or torch.all(di2s < eps):
            break

        # Select item with maximum marginal gain
        j = torch.argmax(di2s).item()
        selected_items.append(j)

        # Update cached values
        if len(cis) > 0:
            proj = torch.matmul(cis, L[j])    # Shape: [len(selected),]
            cj = (L[j] - torch.matmul(proj, cis)) / torch.sqrt(di2s[j])
        else:
            cj = L[j] / torch.sqrt(di2s[j])

        cis = torch.cat([cis, cj.unsqueeze(0)], dim=0)
        di2s = di2s - cj**2
        di2s[j] = -float('inf')  # prevent re-selection

    return selected_items


def greedy_map_dpp_fast_torch_non_muted(L, k=None, eps=1e-10):
    """
    Fast greedy MAP inference for DPP using Cholesky updates (PyTorch version).
    
    Args:
        L: L-ensemble matrix (torch tensor) [n x n], symmetric and PSD
        k: number of items to select (optional)
        eps: small threshold for stopping
    
    Returns:
        selected_items: list of selected indices
    """
    n = L.shape[0]
    device = L.device

    cis = torch.zeros((0, n), device=device)
    di2s = torch.diag(L).clone().detach()  # Detach to avoid autograd side-effects
    selected_items = []

    # Keep track of selected indices
    already_selected = torch.zeros(n, dtype=torch.bool, device=device)

    while k is None or len(selected_items) < k:
        if len(selected_items) == n or torch.all(di2s < eps):
            break

        j = torch.argmax(di2s).item()
        if di2s[j] < eps:
            break

        selected_items.append(j)
        already_selected[j] = True

        # Update Cholesky cache
        if len(cis) > 0:
            proj = torch.matmul(cis, L[j])  # [len(selected)]
            cj = (L[j] - torch.matmul(proj, cis)) / torch.sqrt(di2s[j])
        else:
            cj = L[j] / torch.sqrt(di2s[j])

        cis = torch.cat([cis, cj.unsqueeze(0)], dim=0)

        # Compute new marginal gains safely
        di2s = di2s - cj ** 2
        di2s[already_selected] = -float('inf')  # prevent re-picking

    return selected_items


def marginal_gain(L,A,i):
    """https://github.com/insuhan/fastdppmap/blob/master/bases/logdet_margin_cg.m"""
    if len(A) == 0:
        return log(L[i, i])
    A = list(A)  # ensure proper indexing
    LAA = L[np.ix_(A, A)]
    LAi = L[np.ix_(A, [i])].flatten()
    Lii = L[i, i]
    gain = Lii - LAi.T @ solve(LAA, LAi)
    return log(max(gain, 1e-10))  # prevent log of non-positive values

def dpp_map_greedy(L):
	"""even dumber version of https://github.com/insuhan/fastdppmap/blob/master/algorithms/greedy_lazy.m"""
	n = len(L)
	Y = set(range(n))
	A = set()
	while True:
		delta = {
			i: marginal_gain(L,A,i)
			for i in Y - A
		}
		best = max(delta, key=delta.get)
		gain = delta[best]
		print(f"i: {i}, gain: {gain}")
		if gain > 0:
			A.add(best)
			# A &= {best}
		else:
			return A
          

def DPP_loss(K, sampled_set, mask):
    D_selected, D_ignored = get_user_feedback(samples=sampled_set, mask=mask)
    logprob_selected_in_sampled = K[D_selected,:][:,D_selected].logdet()
    logprob_selected_i_in_samples = 0
    for i in D_ignored:
        D_selected_i = torch.concat([D_selected, torch.tensor([i])])
        logprob_selected_i_in_samples += D_selected_i.logdet()
    return -logprob_selected_in_sampled*(len(D_ignored)+1) + logprob_selected_i_in_samples

# standard
def get_user_feedback(samples, mask):
    # mask = [True,True,True,True, False, True, False, True,True]
    not_mask = [not elem for elem in mask]
    D_selected= torch.tensor(samples)[mask]
    D_ignored = torch.tensor(samples)[not_mask]
    # L[D_selected,:][:,D_selected]
    return D_selected, D_ignored


def get_user_feedback_qvhighlights(L, samples, relevant_clip_ids, relevant_windows, saliency_scores):
    batch_number = len(samples)
    losses = []
    for i in range(batch_number):
        # calculate K
        L_i = L[i]
        n = L_i.shape[0]
        I = torch.eye(L_i.shape[0], device=L.device) 
        # K_i =  L_i @ torch.linalg.inv(I + L_i)
        K_i = calc_K_from_L(L_i)

        relevant_windows_i = relevant_windows[i]
        relevant_clip_ids_i = relevant_clip_ids[i]
        samples_i = samples[i]
        saliency_scores_i = saliency_scores[i]

        saliency_scores_mean_i = saliency_scores_i.to(torch.float).mean(dim=-1)

        if relevant_clip_ids_i[0] != 0:
            relevant_clip_ids_i = relevant_clip_ids_i[relevant_clip_ids_i.nonzero()][:, 0]
            saliency_scores_mean_i = saliency_scores_mean_i[saliency_scores_mean_i.nonzero()][:, 0]
        else:
            relevant_clip_ids_i = relevant_clip_ids_i[relevant_clip_ids_i.nonzero()][:, 0]
            relevant_clip_ids_i = torch.cat((torch.zeros(1).to(relevant_clip_ids_i.device), relevant_clip_ids_i), 0)
        
        "Assuming no diversity"
        mask_i = torch.zeros_like(torch.tensor(samples_i), dtype=torch.bool)
        max_index = 0
        max_saliency_score  = 0
        for j, s in enumerate(samples_i):
            if s in relevant_clip_ids_i:
                index = torch.where(relevant_clip_ids_i == s)
                if saliency_scores_mean_i[index] > max_saliency_score:
                    max_saliency_score = saliency_scores_mean_i[index]
                    max_index = j
        mask_i[max_index] = torch.tensor(True)

        reward = DPP_loss(K_i, samples_i, mask_i)
        prob = dpp_probability(L_i/L_i.diag().max(), samples_i)
        loss = -reward * torch.log(prob)
        losses.append(loss)

        

        
        # indices = torch.tensor([relevant_clip_ids_i.tolist().index(s) for s in samples_i], device='cuda:2')
        # indices = torch.tensor(
        #             [torch.where(relevant_clip_ids_i == s)[0][0].item() if s in relevant_clip_ids_i else -1 for s in samples_i],
        #             device='cuda:2'
        #         )
        # valid_mask = indices != -1
        # valid_indices = indices[valid_mask]

        # saliency_scores_mean_samples_i = torch.zeros_like(torch.tensor(samples_i)).to(valid_mask.device)
        # mask_i = torch.zeros_like(torch.tensor(samples_i)).to(valid_mask.device)
        # saliency_scores_mean_samples_i[valid_mask] = saliency_scores_mean_i[valid_indices]
        # mask_i[saliency_scores_mean_samples_i.argmax()] = 1

        "Assuming diversity as per the number of relevant windows"
        segment_number = torch.ceil(relevant_windows_i.count_nonzero()/2)

    
    return torch.stack(losses)
        

    print('done')
    pass

def kernel_diff(frame_embeddings, question_embedding):
    diff = frame_embeddings - question_embedding
    number_of_frames = diff.shape[0]
    L = torch.zeros((number_of_frames, number_of_frames))
    for i in range(number_of_frames):
        for j in range(number_of_frames):
            L[i,j] = (diff[i] @ diff[j].T) / ((diff[i]**2).sum() * (diff[j]**2).sum())
    diff2 = torch.sum(diff**2, dim=-1)
    L_prev = (diff @ diff.T) / (diff2 * diff2[:,None])
    L_new = (diff @ diff.T) / (torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)) @ torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True)).T)

def sample_once(L, dpp):
    "Args: L - square matrix with size frame_numbers x frame_numbers"
    while True:
        try:
            samples = dpp.sample_exact()
            if len(samples) == 0: #TODO get rid of this
                continue
            break
        except ValueError as e:
            continue
    #         print(e)
    #     print("...trying again...")
    # print('actual   num frames =', len(samples))
    samples.sort()
    prob_samples = dpp_probability(L, samples)
    return samples, prob_samples

def process_single_dpp(L, do_L_norm=False):
    "Args: L - square matrix with size frame_numbers x frame_numbers"
    if do_L_norm:
        L_norm = L / L.diag().max()
        dpp = FiniteDPP("likelihood", L = L_norm.detach().cpu())
        samples, prob_samples = sample_once(L_norm, dpp)
    else:
        dpp = FiniteDPP("likelihood", L = L.detach().cpu())
        samples, prob_samples = sample_once(L, dpp)
        L_norm = None
    return L_norm, dpp, samples, prob_samples

def kernel_simple(frame_embeddings, question_embedding, normalize_inputs=True):
    """Args:
        frame_embeddings: Tensor of shape [n_images, 768]
        question_embedding: Tensor of shape [768]"""
    
    # Normalize embeddings
    if normalize_inputs:
        frame_embeddings = torch.nn.functional.normalize(frame_embeddings, dim=-1)
        question_embedding = torch.nn.functional.normalize(question_embedding, dim=-1)

    # Calculate quality scores (relevance to query)
    quality_scores = frame_embeddings @ question_embedding
    
    # Compute similarity matrix
    similarity = frame_embeddings @ frame_embeddings.T

    # Create the L-ensemble kernel: L(i,j) = q_i * q_j * S(i,j)
    L_kernel = quality_scores[:, None] * quality_scores[None, :] * similarity

    # n = len(image_embeddings)
    # L_kernel = torch.zeros((n, n))    
    # for i in range(n):
    #     for j in range(n):
    #         L_kernel[i, j] = quality_scores[i] * quality_scores[j] * similarity[i, j]
    return L_kernel

def dpp_selection(vision_embeddings, question_embedding):
    """Args:
        frame_embeddings: Tensor of shape [n_images, 768]
        question_embedding: Tensor of shape [768]
    """
    L = kernel_simple(vision_embeddings, question_embedding)
    L_norm, dpp, samples, prob_samples = process_single_dpp(L)
    return L, L_norm, dpp, samples

def kernel_simple_batched(frame_embeddings, question_embedding, normalize_inputs=True):
    """Args:
        frame_embeddings: Tensor of shape [batch_size, n_images, 768]
        question_embedding: Tensor of shape [batch_size, 768]
    """
    
    # Normalize embeddings
    if normalize_inputs:
        frame_embeddings = torch.nn.functional.normalize(frame_embeddings, dim=-1)
        question_embedding = torch.nn.functional.normalize(question_embedding, dim=-1)

    # Calculate quality scores (relevance to query)
    # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
    quality_scores = torch.bmm(frame_embeddings, question_embedding[:, : , None])[..., 0] # [batch_size, n_images]
    
    # Compute similarity matrix
    similarity = frame_embeddings @ frame_embeddings.permute(0, 2, 1) # [batch_size, n_images, n_images]

    # Create the L-ensemble kernel: L(i,j) = q_i * q_j * S(i,j)
    L_kernel = quality_scores[:, :, None] * quality_scores[:, None, :] * similarity # [batch_size, n_images, n_images]

    return L_kernel

def dpp_generation_batched(vision_embeddings, question_embedding):
    """Args:
        frame_embeddings: Tensor of shape [batch_size, n_images, 768]
        question_embedding: Tensor of shape [batch_size, 768]
    """
    L = kernel_simple_batched(vision_embeddings, question_embedding)

    L_norms = []
    dpps = []
    sampless = []

    for L_in in L:
        L_norm, dpp, samples, prob_samples = process_single_dpp(L_in)
        L_norms.append(L_norm)
        dpps.append(dpp)
        sampless.append(samples)
        
    return L, torch.stack(L_norms), dpps, sampless

    L_list = [L[i] for i in range(L.shape[0])]    
    results = parallel_apply([functools.partial(process_single_dpp, L_in) for L_in in L_list])
    
    L_norm, dpp, samples, prob_samples = zip(*results)
    return L, L_norm, dpp, samples, prob_samples


def calc_K_from_L(L):
    K = torch.linalg.solve(L + torch.eye(L.shape[-1], device=L.device), L)
    return K


def expected_num_frames(*, K=None, L=None):
    if K is None: K = calc_K_from_L(L)
    else: assert L is None
    return K.trace()

def dpp_probability(L, samples):
    "Args: L - square matrix with size frame_numbers x frame_numbers which was to define the dpp from which the samples were drawn"
    L_S = L[samples, :][:, samples] # Extract submatrix
    try:
        prob = torch.det(L_S)
    except Exception as e:
        print(f"Error calculating determinant: {e}")
        prob = torch.tensor(1e-6) # To prevent log(0)
    return prob
