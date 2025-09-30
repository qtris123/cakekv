"""
Utility functions for calculating attention matrix metrics.

This module provides functions to analyze multi-head attention matrices during the prefilling stage.
The attention matrices are structured as: [ [3x2059x2059 matrix], [3x2059x2059 matrix], ... ]
where each matrix represents one layer with 3 attention heads of size 2059x2059.
"""

import numpy as np
import torch
import torch.nn.functional as F


def calculate_column_variance_sum(attention_matrix, mask=None):
    """
    Calculate the sum of variance of columns for each attention head.
    
    Args:
        attention_matrix: numpy array of shape (num_heads, seq_len, seq_len) or torch tensor
        mask: optional mask array/tensor of shape (seq_len, seq_len) where True/1 indicates valid positions
        
    Returns:
        numpy array of shape (num_heads,) containing sum of column variances for each head
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    num_heads, seq_len, _ = attention_matrix.shape
    column_variance_sums = np.zeros(num_heads)
    
    for head_idx in range(num_heads):
        head_attention = attention_matrix[head_idx]  # Shape: (seq_len, seq_len)
        
        if mask is not None:
            # Apply mask - set masked positions to NaN so they're ignored in variance calculation
            masked_attention = head_attention.copy()
            masked_attention[~mask] = np.nan
        else:
            masked_attention = head_attention
        
        # Calculate variance for each column, ignoring NaN values
        column_variances = np.nanvar(masked_attention, axis=0)
        
        # Sum the variances across all columns
        column_variance_sums[head_idx] = np.nansum(column_variances)
    
    return column_variance_sums


def calculate_row_cross_entropy(attention_matrix, mask=None):
    """
    Calculate cross-entropy loss for each row in each attention head.
    
    Args:
        attention_matrix: numpy array of shape (num_heads, seq_len, seq_len) or torch tensor
        mask: optional mask array/tensor of shape (seq_len, seq_len) where True/1 indicates valid positions
        
    Returns:
        numpy array of shape (num_heads, seq_len) containing cross-entropy loss for each row in each head
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    num_heads, seq_len, _ = attention_matrix.shape
    cross_entropy_losses = np.zeros((num_heads, seq_len))
    
    for head_idx in range(num_heads):
        head_attention = attention_matrix[head_idx]  # Shape: (seq_len, seq_len)
        
        for row_idx in range(seq_len):
            row_attention = head_attention[row_idx]  # Shape: (seq_len,)
            
            if mask is not None:
                # Get valid positions for this row
                valid_mask = mask[row_idx]
                if not np.any(valid_mask):
                    # If no valid positions, set cross-entropy to 0
                    cross_entropy_losses[head_idx, row_idx] = 0.0
                    continue
                
                # Extract valid attention weights
                valid_attention = row_attention[valid_mask]
                
                # Normalize to get probabilities (softmax)
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                attention_probs = valid_attention - np.max(valid_attention)  # Numerical stability
                attention_probs = np.exp(attention_probs)
                attention_probs = attention_probs / (np.sum(attention_probs) + epsilon)
                
                # Calculate cross-entropy loss assuming uniform distribution as target
                # This measures how far the attention distribution is from uniform
                uniform_prob = 1.0 / len(valid_attention)
                cross_entropy = -np.sum(attention_probs * np.log(attention_probs + epsilon))
                
            else:
                # No masking - use all positions
                # Normalize to get probabilities (softmax)
                epsilon = 1e-8
                attention_probs = row_attention - np.max(row_attention)  # Numerical stability
                attention_probs = np.exp(attention_probs)
                attention_probs = attention_probs / (np.sum(attention_probs) + epsilon)
                
                # Calculate cross-entropy loss assuming uniform distribution as target
                uniform_prob = 1.0 / len(row_attention)
                cross_entropy = -np.sum(attention_probs * np.log(attention_probs + epsilon))
            
            cross_entropy_losses[head_idx, row_idx] = cross_entropy
    
    return cross_entropy_losses


def calculate_attention_metrics(prefil_full_all2, mask=None):
    """
    Calculate both metrics for the entire prefilling attention data structure.
    
    Args:
        prefil_full_all2: List of attention matrices, each of shape (3, 2059, 2059)
        mask: optional mask array/tensor of shape (2059, 2059) where True/1 indicates valid positions
        
    Returns:
        dict containing:
            - 'column_variance_sums': List of arrays, one per layer, containing sum of column variances for each head
            - 'row_cross_entropy': List of arrays, one per layer, containing cross-entropy for each row in each head
    """
    results = {
        'column_variance_sums': [],
        'row_cross_entropy': []
    }
    
    for layer_idx, attention_matrix in enumerate(prefil_full_all2):
        print(f"Processing layer {layer_idx}...")
        
        # Calculate column variance sums for this layer
        col_var_sums = calculate_column_variance_sum(attention_matrix, mask)
        results['column_variance_sums'].append(col_var_sums)
        
        # Calculate row cross-entropy for this layer
        row_cross_ent = calculate_row_cross_entropy(attention_matrix, mask)
        results['row_cross_entropy'].append(row_cross_ent)
    
    return results


def create_causal_mask(seq_len):
    """
    Create a causal mask for attention matrices to handle autoregressive generation.
    
    Args:
        seq_len: Length of the sequence
        
    Returns:
        numpy array of shape (seq_len, seq_len) where True indicates valid positions
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


def create_padding_mask(lengths, max_len):
    """
    Create a padding mask for variable length sequences.
    
    Args:
        lengths: List or array of sequence lengths
        max_len: Maximum sequence length
        
    Returns:
        numpy array of shape (batch_size, max_len) where True indicates valid positions
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    print("Creating example attention matrices...")
    
    # Create dummy attention matrices (2 layers, 3 heads each, 100x100 for demo)
    num_layers = 2
    num_heads = 3
    seq_len = 100
    
    prefil_full_all2 = []
    for layer in range(num_layers):
        # Create random attention matrices
        attention_matrix = np.random.rand(num_heads, seq_len, seq_len)
        # Apply softmax to make them proper attention weights
        for head in range(num_heads):
            attention_matrix[head] = F.softmax(torch.tensor(attention_matrix[head]), dim=-1).numpy()
        prefil_full_all2.append(attention_matrix)
    
    # Create a causal mask
    mask = create_causal_mask(seq_len)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_attention_metrics(prefil_full_all2, mask)
    
    print(f"Column variance sums shape: {[arr.shape for arr in metrics['column_variance_sums']]}")
    print(f"Row cross-entropy shape: {[arr.shape for arr in metrics['row_cross_entropy']]}")
    
    print("Example column variance sums for first layer:")
    print(metrics['column_variance_sums'][0])
    
    print("Example row cross-entropy for first layer (first 5 rows, all heads):")
    print(metrics['row_cross_entropy'][0][:, :5])
