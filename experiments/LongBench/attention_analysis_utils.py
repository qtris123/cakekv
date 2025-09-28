import numpy as np
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    """Utility class for analyzing attention patterns and CAKE behavior"""
    
    def __init__(self, data_dir: str = "./attention_analysis"):
        self.data_dir = Path(data_dir)
        self.analysis_results = {}
        
    def load_attention_data(self, sample_idx: int) -> Dict[str, Any]:
        """Load attention data from pickle file"""
        data_file = self.data_dir / f"attention_data_sample_{sample_idx}.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"Attention data file not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            return pickle.load(f)
    
    def analyze_attention_distribution(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention weight distributions across layers and steps"""
        results = {
            'layer_stats': {},
            'step_stats': {},
            'overall_stats': {}
        }
        
        # Analyze each layer
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            layer_stats = {
                'mean_attention': [],
                'std_attention': [],
                'max_attention': [],
                'min_attention': [],
                'entropy_attention': []
            }
            
            for step_idx, attention_weights in enumerate(layer_attentions):
                if isinstance(attention_weights, torch.Tensor):
                    attention_weights = attention_weights.numpy()
                
                # Flatten attention weights
                flat_weights = attention_weights.flatten()
                
                # Calculate statistics
                layer_stats['mean_attention'].append(np.mean(flat_weights))
                layer_stats['std_attention'].append(np.std(flat_weights))
                layer_stats['max_attention'].append(np.max(flat_weights))
                layer_stats['min_attention'].append(np.min(flat_weights))
                
                # Calculate entropy
                # Normalize to get probabilities
                probs = flat_weights / np.sum(flat_weights)
                probs = probs[probs > 0]  # Remove zeros
                entropy = -np.sum(probs * np.log2(probs))
                layer_stats['entropy_attention'].append(entropy)
            
            results['layer_stats'][layer_idx] = layer_stats
        
        # Calculate overall statistics
        all_means = []
        all_stds = []
        all_entropies = []
        
        for layer_stats in results['layer_stats'].values():
            all_means.extend(layer_stats['mean_attention'])
            all_stds.extend(layer_stats['std_attention'])
            all_entropies.extend(layer_stats['entropy_attention'])
        
        results['overall_stats'] = {
            'mean_attention': np.mean(all_means),
            'std_attention': np.mean(all_stds),
            'mean_entropy': np.mean(all_entropies),
            'std_entropy': np.std(all_entropies)
        }
        
        return results
    
    def analyze_cake_compression_effectiveness(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how effectively CAKE compresses attention patterns"""
        results = {
            'compression_ratios': {},
            'information_preservation': {},
            'eviction_patterns': {}
        }
        
        # Analyze compression ratios
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            if layer_idx in attention_data['layer_budgets']:
                budgets = attention_data['layer_budgets'][layer_idx]
                compression_ratios = []
                
                for step_idx, attention_weights in enumerate(layer_attentions):
                    if step_idx < len(budgets):
                        original_size = attention_weights.shape[-1]  # Sequence length
                        compressed_size = budgets[step_idx]
                        if compressed_size > 0:
                            compression_ratio = original_size / compressed_size
                            compression_ratios.append(compression_ratio)
                
                results['compression_ratios'][layer_idx] = {
                    'mean': np.mean(compression_ratios) if compression_ratios else 0,
                    'std': np.std(compression_ratios) if compression_ratios else 0,
                    'ratios': compression_ratios
                }
        
        # Analyze information preservation using entropy
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            if layer_idx in attention_data['layer_entropy_scores']:
                entropy_scores = attention_data['layer_entropy_scores'][layer_idx]
                variance_scores = attention_data['layer_variance_scores'][layer_idx]
                
                results['information_preservation'][layer_idx] = {
                    'mean_entropy': np.mean(entropy_scores),
                    'std_entropy': np.std(entropy_scores),
                    'mean_variance': np.mean(variance_scores),
                    'std_variance': np.std(variance_scores),
                    'entropy_trend': np.polyfit(range(len(entropy_scores)), entropy_scores, 1)[0] if len(entropy_scores) > 1 else 0
                }
        
        return results
    
    def analyze_layer_attention_patterns(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention patterns across different layers"""
        results = {
            'layer_similarity': {},
            'attention_focus': {},
            'layer_evolution': {}
        }
        
        layers = list(attention_data['layer_attention_scores'].keys())
        
        # Calculate layer similarity
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers[i+1:], i+1):
                similarities = []
                
                # Compare attention patterns across steps
                min_steps = min(len(attention_data['layer_attention_scores'][layer1]),
                              len(attention_data['layer_attention_scores'][layer2]))
                
                for step_idx in range(min_steps):
                    attn1 = attention_data['layer_attention_scores'][layer1][step_idx]
                    attn2 = attention_data['layer_attention_scores'][layer2][step_idx]
                    
                    if isinstance(attn1, torch.Tensor):
                        attn1 = attn1.numpy()
                    if isinstance(attn2, torch.Tensor):
                        attn2 = attn2.numpy()
                    
                    # Flatten and normalize
                    flat1 = attn1.flatten()
                    flat2 = attn2.flatten()
                    
                    # Calculate cosine similarity
                    similarity = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
                    similarities.append(similarity)
                
                results['layer_similarity'][f"{layer1}_{layer2}"] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'similarities': similarities
                }
        
        # Analyze attention focus (how concentrated attention is)
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            focus_scores = []
            
            for step_idx, attention_weights in enumerate(layer_attentions):
                if isinstance(attention_weights, torch.Tensor):
                    attention_weights = attention_weights.numpy()
                
                # Calculate attention focus (inverse of entropy)
                flat_weights = attention_weights.flatten()
                probs = flat_weights / np.sum(flat_weights)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                focus_score = 1 / (entropy + 1e-10)  # Add small epsilon to avoid division by zero
                focus_scores.append(focus_score)
            
            results['attention_focus'][layer_idx] = {
                'mean_focus': np.mean(focus_scores),
                'std_focus': np.std(focus_scores),
                'focus_scores': focus_scores
            }
        
        return results
    
    def analyze_eviction_patterns(self, attention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CAKE eviction patterns and their effectiveness"""
        results = {
            'eviction_frequency': {},
            'eviction_impact': {},
            'budget_utilization': {}
        }
        
        # Analyze eviction frequency
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            if layer_idx in attention_data['layer_budgets']:
                budgets = attention_data['layer_budgets'][layer_idx]
                eviction_frequency = []
                
                for step_idx, attention_weights in enumerate(layer_attentions):
                    if step_idx < len(budgets):
                        original_size = attention_weights.shape[-1]
                        budget = budgets[step_idx]
                        if budget > 0 and original_size > budget:
                            eviction_frequency.append(1)  # Eviction occurred
                        else:
                            eviction_frequency.append(0)  # No eviction
                
                results['eviction_frequency'][layer_idx] = {
                    'frequency': np.mean(eviction_frequency),
                    'total_evictions': np.sum(eviction_frequency),
                    'eviction_steps': eviction_frequency
                }
        
        # Analyze budget utilization
        for layer_idx, layer_attentions in attention_data['layer_attention_scores'].items():
            if layer_idx in attention_data['layer_budgets']:
                budgets = attention_data['layer_budgets'][layer_idx]
                utilization_ratios = []
                
                for step_idx, attention_weights in enumerate(layer_attentions):
                    if step_idx < len(budgets):
                        actual_size = attention_weights.shape[-1]
                        budget = budgets[step_idx]
                        if budget > 0:
                            utilization = actual_size / budget
                            utilization_ratios.append(utilization)
                
                results['budget_utilization'][layer_idx] = {
                    'mean_utilization': np.mean(utilization_ratios) if utilization_ratios else 0,
                    'std_utilization': np.std(utilization_ratios) if utilization_ratios else 0,
                    'utilization_ratios': utilization_ratios
                }
        
        return results
    
    def generate_analysis_report(self, sample_idx: int, output_dir: str = "./attention_analysis/reports"):
        """Generate a comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load attention data
        attention_data = self.load_attention_data(sample_idx)
        
        # Perform analyses
        attention_dist = self.analyze_attention_distribution(attention_data)
        compression_effectiveness = self.analyze_cake_compression_effectiveness(attention_data)
        layer_patterns = self.analyze_layer_attention_patterns(attention_data)
        eviction_patterns = self.analyze_eviction_patterns(attention_data)
        
        # Combine results
        report = {
            'sample_idx': sample_idx,
            'attention_distribution': attention_dist,
            'compression_effectiveness': compression_effectiveness,
            'layer_patterns': layer_patterns,
            'eviction_patterns': eviction_patterns,
            'summary': self._generate_summary(attention_dist, compression_effectiveness, layer_patterns, eviction_patterns)
        }
        
        # Save report
        report_file = Path(output_dir) / f"analysis_report_sample_{sample_idx}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_analysis_plots(report, sample_idx, output_dir)
        
        print(f"Analysis report generated: {report_file}")
        return report
    
    def _generate_summary(self, attention_dist, compression_effectiveness, layer_patterns, eviction_patterns):
        """Generate a summary of the analysis"""
        summary = {
            'total_layers': len(attention_dist['layer_stats']),
            'overall_attention_entropy': attention_dist['overall_stats']['mean_entropy'],
            'compression_effectiveness': {},
            'layer_diversity': {},
            'eviction_efficiency': {}
        }
        
        # Compression effectiveness summary
        if compression_effectiveness['compression_ratios']:
            avg_compression = np.mean([stats['mean'] for stats in compression_effectiveness['compression_ratios'].values()])
            summary['compression_effectiveness']['average_compression_ratio'] = avg_compression
        
        # Layer diversity summary
        if layer_patterns['layer_similarity']:
            avg_similarity = np.mean([stats['mean_similarity'] for stats in layer_patterns['layer_similarity'].values()])
            summary['layer_diversity']['average_layer_similarity'] = avg_similarity
        
        # Eviction efficiency summary
        if eviction_patterns['eviction_frequency']:
            avg_eviction_freq = np.mean([stats['frequency'] for stats in eviction_patterns['eviction_frequency'].values()])
            summary['eviction_efficiency']['average_eviction_frequency'] = avg_eviction_freq
        
        return summary
    
    def _generate_analysis_plots(self, report, sample_idx, output_dir):
        """Generate analysis plots"""
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Attention entropy across layers
        if 'layer_stats' in report['attention_distribution']:
            layers = list(report['attention_distribution']['layer_stats'].keys())
            entropies = [report['attention_distribution']['layer_stats'][layer]['entropy_attention'] 
                        for layer in layers]
            
            plt.figure(figsize=(12, 8))
            for i, (layer, entropy) in enumerate(zip(layers, entropies)):
                plt.plot(entropy, label=f'Layer {layer}', marker='o')
            
            plt.xlabel('Inference Step')
            plt.ylabel('Attention Entropy')
            plt.title('Attention Entropy Evolution Across Layers')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"attention_entropy_evolution_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Compression ratios
        if 'compression_ratios' in report['compression_effectiveness']:
            layers = list(report['compression_effectiveness']['compression_ratios'].keys())
            ratios = [report['compression_effectiveness']['compression_ratios'][layer]['mean'] 
                     for layer in layers]
            
            plt.figure(figsize=(10, 6))
            plt.bar(layers, ratios)
            plt.xlabel('Layer')
            plt.ylabel('Average Compression Ratio')
            plt.title('Compression Ratios Across Layers')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / f"compression_ratios_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Layer similarity heatmap
        if 'layer_similarity' in report['layer_patterns']:
            similarity_data = report['layer_patterns']['layer_similarity']
            layers = sorted(set([key.split('_')[0] for key in similarity_data.keys()] + 
                              [key.split('_')[1] for key in similarity_data.keys()]))
            
            similarity_matrix = np.eye(len(layers))
            for key, stats in similarity_data.items():
                layer1, layer2 = key.split('_')
                i, j = layers.index(layer1), layers.index(layer2)
                similarity_matrix[i, j] = stats['mean_similarity']
                similarity_matrix[j, i] = stats['mean_similarity']
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, 
                       xticklabels=layers, 
                       yticklabels=layers, 
                       annot=True, 
                       cmap='viridis')
            plt.title('Layer Similarity Matrix')
            plt.savefig(plots_dir / f"layer_similarity_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Example usage of the AttentionAnalyzer"""
    analyzer = AttentionAnalyzer("./attention_analysis")
    
    try:
        # Generate analysis report for sample 0
        report = analyzer.generate_analysis_report(0)
        
        print("Analysis completed successfully!")
        print(f"Summary: {report['summary']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the attention analysis script first to generate data.")


if __name__ == "__main__":
    main()
