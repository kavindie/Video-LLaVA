import torch
import sklearn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dppy.finite_dpps import FiniteDPP

def compare_embeds(video_vector, image_vectors):
    cosine_similarity = torch.nn.functional.cosine_similarity(video_vector, image_vectors)
    softmax_op = torch.nn.Softmax(dim=0)
    pdf = softmax_op(cosine_similarity)
    

    best_x, best_error, best_lambda = solve_with_lasso(image_vectors.to('cpu').T, video_vector.to('cpu'))
    print(f"Argmax: {pdf.argmax().item()}, Best x: {best_x}")
    return pdf, best_x


def solve_with_lasso(A, y, alpha_range=None):
    """
    Use Lasso regression with automatic alpha tuning to find an approximate sparse solution.
    
    Parameters:
    A: np.ndarray of shape [m, n] -  frame embeddings
    y: np.ndarray of shape [m] -  video embedding
    alpha_range: list - custom range of alpha values to try
    
    Returns:
    x: np.ndarray - binary solution vector
    error: float - cosine distance between λAx and y
    lambda_scale: float - scaling factor λ
    """
    from sklearn.linear_model import Lasso
    if alpha_range is None:
        # Try a wide range of alpha values, exponentially spaced
        alpha_range = np.logspace(-4, 0, 20)
    
    best_x = None
    best_error = float('inf')
    best_lambda = None
    
    for alpha in alpha_range:
        # Solve using Lasso regression
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
        lasso.fit(A, y)
        
        # Skip if all coefficients are zero
        if np.all(lasso.coef_ == 0):
            continue
            
        # Try different thresholds for binarization
        percentiles = [50, 60, 70, 80, 90]
        for p in percentiles:
            threshold = np.percentile(lasso.coef_[lasso.coef_ > 0], p) if np.any(lasso.coef_ > 0) else 0
            x = np.where(lasso.coef_ > threshold, 1, 0)
            
            # Skip if all zeros
            if np.sum(x) == 0:
                continue
                
            # Calculate error and scaling factor
            Ax = A @ x
            lambda_scale = 1.0 / (np.linalg.norm(Ax) + 1e-10)
            Ax_normalized = Ax * lambda_scale
            error = 1.0 - np.dot(Ax_normalized, y)
            
            if error < best_error:
                best_error = error
                best_x = x
                best_lambda = lambda_scale
    
    if best_x is None:
        raise ValueError("Could not find non-zero solution with Lasso")
        
    return best_x, best_error, best_lambda


def DPP(B, lambda_param = 0.01):
    L = B @ B.T
    L = L.cpu()

    dpp = FiniteDPP("likelihood", L = lambda_param * L)
    return dpp.sample_exact()
     




    """feature_embeddings - torch.tensor of shape [num_frames, embedding_dim]
       query_embedding - torch.tensor of shape [embedding_dim]"""
    # L matrix
    # L = feature_embeddings @ feature_embeddings.T

    # Compute pairwise frame similarity (diversity) x1@x2 / (||x1|| * ||x2||)
    feature_embeddings_norm = torch.nn.functional.normalize(feature_embeddings, p=2, dim=1)
    S = torch.mm(feature_embeddings_norm, feature_embeddings_norm.t())
    # numpy version   S = cosine_similarity(feature_embeddings.cpu())


    # Compute relevance scores (cosine similarity with query)
    query_embedding_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    relevance_scores = torch.mv(feature_embeddings_norm, query_embedding_norm)
    # Normalize relevance scores to [0,1]
    relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min())
    
    # Modify L-matrix: Balance diversity and relevance
    # Set lambda parameter for tradeoff
    # Compute outer product of relevance scores and adjust with lambda
    # First, compute outer product
    relevance_outer = torch.ger(relevance_scores, relevance_scores)
    # Raise outer product to the power of lambda_param (element-wise)
    relevance_weight = relevance_outer ** lambda_param

    # Modify L-matrix: balance diversity (S) and relevance (relevance_weight)
    L = S * relevance_weight
    L_np = L.detach().cpu().numpy()
    # Define and sample from DPP
    dpp = FiniteDPP("likelihood", L=L_np)
    dpp.sample_exact()
    return dpp
    summary = [frames[i] for i in dpp.list_of_samples[0]]

class DPPLasso(torch.nn.Module):
    def __init__(self, input_dim, kernel_type='linear', alpha=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.kernel_type = kernel_type
        
        # Learnable quality term
        self.quality = torch.nn.Parameter(torch.ones(input_dim))
        
    def compute_kernel(self, X):
        """Compute the DPP kernel matrix."""
        if self.kernel_type == 'linear':
            # Similarity matrix
            S = torch.mm(X, X.t())
            # Quality terms
            Q = self.quality.unsqueeze(0) * self.quality.unsqueeze(1)
            # Final kernel
            L = S * Q
        return L
        
    def dpp_log_probability(self, L, selected_items):
        """Compute log probability of a particular subset under the DPP."""
        # Get selected submatrix
        mask = selected_items.unsqueeze(0) * selected_items.unsqueeze(1)
        L_selected = L * mask
        
        # Add small diagonal term for numerical stability
        L_selected = L_selected + torch.eye(L.size(0)).to(L.device) * 1e-5
        
        # Compute log determinant
        eigenvalues = torch.linalg.eigvalsh(L_selected)
        log_det = torch.sum(torch.log(torch.clamp(eigenvalues, min=1e-40)))
        
        return log_det
        
    def forward(self, X, w):
        """
        X: input features [batch_size, input_dim]
        w: Lasso weights [input_dim]
        """
        # Compute DPP kernel
        L = self.compute_kernel(X)
        
        # Convert Lasso weights to binary selections via sigmoid
        selections = torch.sigmoid(w)
        
        # DPP log probability
        dpp_log_prob = self.dpp_log_probability(L, selections)
        
        # Lasso term
        lasso_term = (1.0 / (2 * X.size(0))) * torch.norm(torch.mm(X, w.unsqueeze(1)), p=2) + self.alpha * torch.norm(w, p=1)
        
        # Combine DPP and Lasso terms
        total_loss = lasso_term - dpp_log_prob
        
        return total_loss, selections