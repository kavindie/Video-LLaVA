from dppy.finite_dpps import FiniteDPP
import torch

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

def kernel_simple(frame_embeddings, question_embedding):
    """Args:
        frame_embeddings: Tensor of shape [n_images, 768]
        question_embedding: Tensor of shape [768]"""
    
    # Normalize embeddings
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

def kernel_simple_batched(frame_embeddings, question_embedding):
    """Args:
        frame_embeddings: Tensor of shape [batch_size, n_images, 768]
        question_embedding: Tensor of shape [batch_size, 768]
    """
    
    # Normalize embeddings
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
