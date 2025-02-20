import torch
import sklearn
import numpy as np


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
    A: np.ndarray of shape [m, n] - normalized frame embeddings
    y: np.ndarray of shape [m] - normalized video embedding
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
