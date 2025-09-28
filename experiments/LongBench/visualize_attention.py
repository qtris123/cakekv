import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class AttentionVisualizer:
    def __init__(self, data_dir: str = "./attention_analysis"):
        self.data_dir = Path(data_dir)
        self.attention_data = None
        
    def load_attention_data(self, sample_idx: int):
        """Load attention data from pickle file"""
        data_file = self.data_dir / f"attention_data_sample_{sample_idx}.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"Attention data file not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            self.attention_data = pickle.load(f)
        
        print(f"Loaded attention data for sample {sample_idx}")
        print(f"Number of layers: {len(self.attention_data['layer_attention_scores'])}")
        print(f"Number of inference steps: {len(self.attention_data['inference_steps'])}")
        
    def plot_attention_heatmap(self, layer_idx: int, step_idx: int = 0, head_idx: int = 0, 
                              save_path: Optional[str] = None, max_tokens: int = 50):
        """Plot attention heatmap for a specific layer, step, and head"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layer_idx not in self.attention_data['layer_attention_scores']:
            raise ValueError(f"Layer {layer_idx} not found in attention data")
        
        if step_idx >= len(self.attention_data['layer_attention_scores'][layer_idx]):
            raise ValueError(f"Step {step_idx} not found for layer {layer_idx}")
        
        attention_weights = self.attention_data['layer_attention_scores'][layer_idx][step_idx]
        
        # Select specific head and limit tokens
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0, head_idx, :max_tokens, :max_tokens]
        elif attention_weights.dim() == 3:  # [heads, seq, seq]
            attention_weights = attention_weights[head_idx, :max_tokens, :max_tokens]
        else:
            attention_weights = attention_weights[:max_tokens, :max_tokens]
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_weights, 
                   cmap='Blues', 
                   cbar=True,
                   square=True,
                   xticklabels=False,
                   yticklabels=False)
        
        plt.title(f'Attention Heatmap - Layer {layer_idx}, Step {step_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_attention_evolution(self, layer_idx: int, head_idx: int = 0, 
                                position_idx: int = -1, save_path: Optional[str] = None):
        """Plot how attention to a specific position evolves across steps"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layer_idx not in self.attention_data['layer_attention_scores']:
            raise ValueError(f"Layer {layer_idx} not found in attention data")
        
        steps = []
        attention_values = []
        
        for step_idx, attention_weights in enumerate(self.attention_data['layer_attention_scores'][layer_idx]):
            if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
                attn = attention_weights[0, head_idx, position_idx, :]
            elif attention_weights.dim() == 3:  # [heads, seq, seq]
                attn = attention_weights[head_idx, position_idx, :]
            else:
                attn = attention_weights[position_idx, :]
            
            if isinstance(attn, torch.Tensor):
                attn = attn.numpy()
            
            steps.append(step_idx)
            attention_values.append(attn)
        
        # Create subplot for each step
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (step, attn) in enumerate(zip(steps[:4], attention_values[:4])):
            if i >= len(axes):
                break
            axes[i].plot(attn)
            axes[i].set_title(f'Step {step}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Attention Weight')
            axes[i].grid(True)
        
        plt.suptitle(f'Attention Evolution - Layer {layer_idx}, Head {head_idx}, Position {position_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_layer_comparison(self, step_idx: int = 0, head_idx: int = 0, 
                             layers_to_plot: Optional[List[int]] = None, 
                             save_path: Optional[str] = None):
        """Compare attention patterns across different layers"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layers_to_plot is None:
            layers_to_plot = list(self.attention_data['layer_attention_scores'].keys())[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, layer_idx in enumerate(layers_to_plot):
            if i >= len(axes):
                break
                
            if layer_idx not in self.attention_data['layer_attention_scores']:
                continue
                
            if step_idx >= len(self.attention_data['layer_attention_scores'][layer_idx]):
                continue
            
            attention_weights = self.attention_data['layer_attention_scores'][layer_idx][step_idx]
            
            # Select specific head
            if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
                attention_weights = attention_weights[0, head_idx, :, :]
            elif attention_weights.dim() == 3:  # [heads, seq, seq]
                attention_weights = attention_weights[head_idx, :, :]
            
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.numpy()
            
            sns.heatmap(attention_weights, 
                       cmap='Blues', 
                       cbar=True,
                       square=True,
                       ax=axes[i],
                       xticklabels=False,
                       yticklabels=False)
            
            axes[i].set_title(f'Layer {layer_idx}')
        
        plt.suptitle(f'Layer Comparison - Step {step_idx}, Head {head_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_entropy_variance_evolution(self, layer_idx: int, save_path: Optional[str] = None):
        """Plot entropy and variance scores evolution across steps"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layer_idx not in self.attention_data['layer_entropy_scores']:
            raise ValueError(f"Layer {layer_idx} not found in entropy data")
        
        steps = list(range(len(self.attention_data['layer_entropy_scores'][layer_idx])))
        entropy_scores = self.attention_data['layer_entropy_scores'][layer_idx]
        variance_scores = self.attention_data['layer_variance_scores'][layer_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot entropy
        ax1.plot(steps, entropy_scores, 'b-o', label='Entropy')
        ax1.set_xlabel('Inference Step')
        ax1.set_ylabel('Entropy Score')
        ax1.set_title(f'Entropy Evolution - Layer {layer_idx}')
        ax1.grid(True)
        ax1.legend()
        
        # Plot variance
        ax2.plot(steps, variance_scores, 'r-o', label='Variance')
        ax2.set_xlabel('Inference Step')
        ax2.set_ylabel('Variance Score')
        ax2.set_title(f'Variance Evolution - Layer {layer_idx}')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_preference_scores(self, save_path: Optional[str] = None):
        """Plot preference scores across layers and steps"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        # Collect all preference scores
        layers = []
        steps = []
        scores = []
        
        for layer_idx, layer_scores in self.attention_data['preference_scores'].items():
            for step_idx, score in enumerate(layer_scores):
                layers.append(layer_idx)
                steps.append(step_idx)
                scores.append(score)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Layer': layers,
            'Step': steps,
            'Preference Score': scores
        })
        
        # Create heatmap
        pivot_df = df.pivot(index='Layer', columns='Step', values='Preference Score')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, 
                   cmap='viridis', 
                   cbar=True,
                   annot=True,
                   fmt='.3f')
        
        plt.title('Preference Scores Across Layers and Steps')
        plt.xlabel('Inference Step')
        plt.ylabel('Layer')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_3d_attention(self, layer_idx: int, step_idx: int = 0, head_idx: int = 0, 
                         save_path: Optional[str] = None):
        """Create 3D visualization of attention weights"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layer_idx not in self.attention_data['layer_attention_scores']:
            raise ValueError(f"Layer {layer_idx} not found in attention data")
        
        if step_idx >= len(self.attention_data['layer_attention_scores'][layer_idx]):
            raise ValueError(f"Step {step_idx} not found for layer {layer_idx}")
        
        attention_weights = self.attention_data['layer_attention_scores'][layer_idx][step_idx]
        
        # Select specific head
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0, head_idx, :, :]
        elif attention_weights.dim() == 3:  # [heads, seq, seq]
            attention_weights = attention_weights[head_idx, :, :]
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.numpy()
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=attention_weights)])
        
        fig.update_layout(
            title=f'3D Attention Surface - Layer {layer_idx}, Step {step_idx}, Head {head_idx}',
            scene=dict(
                xaxis_title='Key Position',
                yaxis_title='Query Position',
                zaxis_title='Attention Weight'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def create_interactive_attention_plot(self, layer_idx: int, step_idx: int = 0, 
                                        save_path: Optional[str] = None):
        """Create interactive attention plot with Plotly"""
        if self.attention_data is None:
            raise ValueError("No attention data loaded. Call load_attention_data() first.")
        
        if layer_idx not in self.attention_data['layer_attention_scores']:
            raise ValueError(f"Layer {layer_idx} not found in attention data")
        
        if step_idx >= len(self.attention_data['layer_attention_scores'][layer_idx]):
            raise ValueError(f"Step {step_idx} not found for layer {layer_idx}")
        
        attention_weights = self.attention_data['layer_attention_scores'][layer_idx][step_idx]
        
        # Average across heads if multiple heads
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0].mean(dim=0)  # Average across heads
        elif attention_weights.dim() == 3:  # [heads, seq, seq]
            attention_weights = attention_weights.mean(dim=0)  # Average across heads
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.numpy()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Interactive Attention Heatmap - Layer {layer_idx}, Step {step_idx}',
            xaxis_title='Key Position',
            yaxis_title='Query Position'
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def generate_comprehensive_report(self, sample_idx: int, output_dir: str = "./attention_analysis/plots"):
        """Generate a comprehensive visualization report"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.load_attention_data(sample_idx)
        
        # Get available layers
        layers = list(self.attention_data['layer_attention_scores'].keys())
        
        # 1. Layer comparison
        self.plot_layer_comparison(
            step_idx=0, 
            head_idx=0, 
            layers_to_plot=layers[:4],
            save_path=f"{output_dir}/layer_comparison_sample_{sample_idx}.png"
        )
        
        # 2. Attention heatmaps for different layers
        for layer_idx in layers[:4]:  # Plot first 4 layers
            self.plot_attention_heatmap(
                layer_idx=layer_idx,
                step_idx=0,
                head_idx=0,
                save_path=f"{output_dir}/attention_heatmap_layer_{layer_idx}_sample_{sample_idx}.png"
            )
        
        # 3. Entropy and variance evolution
        for layer_idx in layers[:4]:
            self.plot_entropy_variance_evolution(
                layer_idx=layer_idx,
                save_path=f"{output_dir}/entropy_variance_layer_{layer_idx}_sample_{sample_idx}.png"
            )
        
        # 4. Preference scores
        self.plot_preference_scores(
            save_path=f"{output_dir}/preference_scores_sample_{sample_idx}.png"
        )
        
        # 5. Interactive plots
        for layer_idx in layers[:2]:  # Create interactive plots for first 2 layers
            self.create_interactive_attention_plot(
                layer_idx=layer_idx,
                step_idx=0,
                save_path=f"{output_dir}/interactive_attention_layer_{layer_idx}_sample_{sample_idx}.html"
            )
        
        print(f"Comprehensive report generated in {output_dir}")


def main():
    """Example usage of the AttentionVisualizer"""
    visualizer = AttentionVisualizer("./attention_analysis")
    
    # Load data for sample 0
    try:
        visualizer.load_attention_data(0)
        
        # Generate comprehensive report
        visualizer.generate_comprehensive_report(0)
        
        # Individual plots
        visualizer.plot_attention_heatmap(layer_idx=0, step_idx=0, head_idx=0)
        visualizer.plot_attention_evolution(layer_idx=0, head_idx=0, position_idx=-1)
        visualizer.plot_3d_attention(layer_idx=0, step_idx=0, head_idx=0)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the attention analysis script first to generate data.")


if __name__ == "__main__":
    main()
