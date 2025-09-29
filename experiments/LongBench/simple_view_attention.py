import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_attention_data(file_path):
    """Load attention data from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_attention_heatmap(attention_data, layer_idx=0, head_idx=0, max_tokens=50):
    """Plot attention heatmap for a specific layer and head"""
    if layer_idx not in attention_data:
        print(f"Layer {layer_idx} not found in data")
        return
    
    attention_weights = attention_data[layer_idx]['tensor']
    
    # Select specific head and limit tokens
    if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
        attention_weights = attention_weights[0, head_idx, :max_tokens, :max_tokens]
    elif len(attention_weights.shape) == 3:  # [heads, seq, seq]
        attention_weights = attention_weights[head_idx, :max_tokens, :max_tokens]
    else:
        attention_weights = attention_weights[:max_tokens, :max_tokens]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_weights, 
               cmap='Blues', 
               cbar=True,
               square=True,
               xticklabels=False,
               yticklabels=False)
    
    plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

def plot_layers_comparison(attention_data, layers=[0, 1, 2, 3], head_idx=0, max_tokens=30):
    """Compare attention patterns across different layers"""
    available_layers = list(attention_data.keys())
    layers_to_plot = [layer for layer in layers if layer in available_layers]
    
    if not layers_to_plot:
        print("No valid layers to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, layer_idx in enumerate(layers_to_plot[:4]):
        attention_weights = attention_data[layer_idx]['tensor']
        
        # Select specific head and limit tokens
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0, head_idx, :max_tokens, :max_tokens]
        elif len(attention_weights.shape) == 3:  # [heads, seq, seq]
            attention_weights = attention_weights[head_idx, :max_tokens, :max_tokens]
        else:
            attention_weights = attention_weights[:max_tokens, :max_tokens]
        
        sns.heatmap(attention_weights, 
                   cmap='Blues', 
                   cbar=True,
                   square=True,
                   ax=axes[i],
                   xticklabels=False,
                   yticklabels=False)
        
        axes[i].set_title(f'Layer {layer_idx}')
    
    # Hide unused subplots
    for i in range(len(layers_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Layer Comparison - Head {head_idx}')
    plt.tight_layout()
    plt.show()

def plot_heads_comparison(attention_data, layer_idx=0, heads=[0, 1, 2, 3], max_tokens=30):
    """Compare attention patterns across different heads"""
    if layer_idx not in attention_data:
        print(f"Layer {layer_idx} not found in data")
        return
        
    attention_weights = attention_data[layer_idx]['tensor']
    num_heads = attention_weights.shape[1] if len(attention_weights.shape) > 3 else 1
    
    available_heads = list(range(min(num_heads, max(heads) + 1)))
    heads_to_plot = [head for head in heads if head in available_heads]
    
    if not heads_to_plot:
        print("No valid heads to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, head_idx in enumerate(heads_to_plot[:4]):
        if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
            head_weights = attention_weights[0, head_idx, :max_tokens, :max_tokens]
        else:
            head_weights = attention_weights[:max_tokens, :max_tokens]
        
        sns.heatmap(head_weights, 
                   cmap='Blues', 
                   cbar=True,
                   square=True,
                   ax=axes[i],
                   xticklabels=False,
                   yticklabels=False)
        
        axes[i].set_title(f'Head {head_idx}')
    
    # Hide unused subplots
    for i in range(len(heads_to_plot), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Heads Comparison - Layer {layer_idx}')
    plt.tight_layout()
    plt.show()

def analyze_attention_data(attention_data):
    """Print basic analysis of attention data"""
    print("Attention Data Analysis:")
    print("=" * 50)
    
    print(f"Total layers: {len(attention_data)}")
    layers = sorted(attention_data.keys())
    
    for layer_idx in layers:
        shape = attention_data[layer_idx]['shape']
        tensor_shape = attention_data[layer_idx]['tensor'].shape
        print(f"Layer {layer_idx}: declared shape {shape}, actual tensor shape {tensor_shape}")
        
        # Basic statistics
        attention_weights = attention_data[layer_idx]['tensor']
        if len(attention_weights.shape) >= 2:
            attn_flat = attention_weights.flatten()
            print(f"  Mean attention: {np.mean(attn_flat):.4f}")
            print(f"  Max attention: {np.max(attn_flat):.4f}")
            print(f"  Min attention: {np.min(attn_flat):.4f}")
            print(f"  Std attention: {np.std(attn_flat):.4f}")
    
    print("\nComputing attention statistics...")
    
    # Overall statistics
    all_attentions = []
    for layer_idx in layers:
        attention_weights = attention_data[layer_idx]['tensor']
        if len(attention_weights.shape) >= 2:
            all_attentions.extend(attention_weights.flatten())
    
    if all_attentions:
        print(f"\nOverall Statistics:")
        print(f"Mean attention: {np.mean(all_attentions):.4f}")
        print(f"Max attention: {np.max(all_attentions):.4f}")
        print(f"Min attention: {np.min(all_attentions):.4f}")
        print(f"Std attention: {np.std(all_attentions):.4f}")

def main():
    """Example usage"""
    # Load data
    data_file = "./attention_data/attention_sample_0.pkl"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Please run the attention collection script first:")
        print("python simple_llama_attention.py --save_attention --max_samples 1")
        return
    
    print("Loading attention data...")
    attention_data = load_attention_data(data_file)
    
    # Analyze the data
    analyze_attention_data(attention_data)
    
    # Plot some visualizations
    print("\nGenerating visualizations...")
    
    # Get available layers
    available_layers = sorted(attention_data.keys())
    
    if not available_layers:
        print("No layers found in attention data")
        return
    
    # Plot attention heatmap for first layer
    plot_attention_heatmap(attention_data, layer_idx=available_layers[0])
    
    # Compare first 4 layers
    layers_to_compare = available_layers[:4]
    plot_layers_comparison(attention_data, layers=layers_to_compare)
    
    # Compare heads in first layer
    attention_shapes = attention_data[available_layers[0]]['tensor'].shape
    if len(attention_shapes) > 3:  # Has heads dimension
        num_heads = attention_shapes[1]
        heads_to_compare = list(range(min(4, num_heads)))
        plot_heads_comparison(attention_data, layer_idx=available_layers[0], heads=heads_to_compare)

if __name__ == "__main__":
    main()
