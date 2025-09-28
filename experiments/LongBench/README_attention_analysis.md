# CAKE Attention Analysis System

This system allows you to investigate attention score distributions over different layers at different inference steps and visualize attention score patterns in CAKE (Compressed Attention with Key-value Eviction).

## Overview

The attention analysis system consists of four main components:

1. **Data Collection** (`pred_cake_attention_analysis.py`) - Modified inference script that captures attention scores
2. **Attention Forward Functions** (`modify_llama_attention_analysis.py`) - Modified attention mechanisms that store attention weights
3. **Visualization** (`visualize_attention.py`) - Comprehensive visualization tools for attention patterns
4. **Analysis Utilities** (`attention_analysis_utils.py`) - Statistical analysis and reporting tools

## Features

### Data Collection
- Captures attention weights at each layer and inference step
- Stores entropy and variance scores used in CAKE compression
- Records preference scores and eviction patterns
- Tracks layer budgets and compression ratios

### Visualization Capabilities
- **Attention Heatmaps**: 2D heatmaps showing attention patterns between tokens
- **3D Attention Surfaces**: Interactive 3D visualizations of attention weights
- **Layer Comparisons**: Side-by-side comparison of attention patterns across layers
- **Attention Evolution**: How attention patterns change across inference steps
- **Interactive Plots**: Plotly-based interactive visualizations
- **Comprehensive Reports**: Automated generation of analysis reports

### Analysis Features
- **Attention Distribution Analysis**: Statistical analysis of attention weight distributions
- **Compression Effectiveness**: Analysis of CAKE compression ratios and information preservation
- **Layer Pattern Analysis**: Similarity analysis between layers and attention focus patterns
- **Eviction Pattern Analysis**: Analysis of CAKE eviction frequency and budget utilization

## Usage

### Quick Start

```bash
# Run complete analysis pipeline
python run_attention_analysis.py --model llama3.1-8b-128k --compress --max_samples 3

# Run with custom CAKE parameters
python run_attention_analysis.py \
    --model llama3.1-8b-128k \
    --compress \
    --cache_size 512 \
    --window_size 16 \
    --tau1 1.2 \
    --tau2 0.8 \
    --gamma 150 \
    --max_samples 5
```

### Step-by-Step Usage

#### 1. Collect Attention Data

```bash
python pred_cake_attention_analysis.py \
    --model llama3.1-8b-128k \
    --compress \
    --save_attention \
    --max_samples 3 \
    --output_dir ./attention_analysis
```

#### 2. Generate Analysis Reports

```python
from attention_analysis_utils import AttentionAnalyzer

analyzer = AttentionAnalyzer("./attention_analysis")
report = analyzer.generate_analysis_report(0)  # Analyze sample 0
```

#### 3. Create Visualizations

```python
from visualize_attention import AttentionVisualizer

visualizer = AttentionVisualizer("./attention_analysis")
visualizer.load_attention_data(0)

# Generate comprehensive report
visualizer.generate_comprehensive_report(0)

# Individual visualizations
visualizer.plot_attention_heatmap(layer_idx=0, step_idx=0, head_idx=0)
visualizer.plot_3d_attention(layer_idx=0, step_idx=0, head_idx=0)
visualizer.create_interactive_attention_plot(layer_idx=0, step_idx=0)
```

## Output Structure

```
attention_analysis/
├── attention_data_sample_0.pkl          # Raw attention data
├── attention_summary_sample_0.json      # Data summary
├── reports/
│   ├── analysis_report_sample_0.json    # Statistical analysis
│   └── plots/
│       ├── attention_entropy_evolution_sample_0.png
│       ├── compression_ratios_sample_0.png
│       └── layer_similarity_sample_0.png
└── plots/
    ├── layer_comparison_sample_0.png
    ├── attention_heatmap_layer_0_sample_0.png
    ├── entropy_variance_layer_0_sample_0.png
    ├── preference_scores_sample_0.png
    └── interactive_attention_layer_0_sample_0.html
```

## Key Visualizations

### 1. Attention Heatmaps
- Show attention weights between query and key positions
- Color intensity represents attention strength
- Useful for understanding which tokens attend to which other tokens

### 2. Layer Comparisons
- Side-by-side comparison of attention patterns across layers
- Reveals how attention patterns evolve through the network
- Helps identify layer-specific attention behaviors

### 3. Attention Evolution
- Shows how attention to specific positions changes across inference steps
- Reveals dynamic attention patterns during generation
- Useful for understanding attention drift

### 4. 3D Attention Surfaces
- Interactive 3D visualization of attention weights
- Provides intuitive understanding of attention patterns
- Allows exploration of attention landscapes

### 5. Compression Analysis
- Shows compression ratios across layers
- Analyzes information preservation during compression
- Evaluates CAKE effectiveness

## Analysis Metrics

### Attention Distribution Metrics
- **Mean Attention**: Average attention weight across all positions
- **Attention Entropy**: Measure of attention distribution uniformity
- **Attention Variance**: Measure of attention weight dispersion
- **Attention Focus**: Inverse of entropy, measures attention concentration

### Compression Metrics
- **Compression Ratio**: Original size / compressed size
- **Information Preservation**: Entropy and variance preservation
- **Budget Utilization**: How effectively cache budgets are used
- **Eviction Frequency**: How often eviction occurs

### Layer Analysis Metrics
- **Layer Similarity**: Cosine similarity between layer attention patterns
- **Attention Focus**: How concentrated attention is in each layer
- **Layer Evolution**: How attention patterns change across steps

## Customization

### Adding New Visualizations

```python
def plot_custom_analysis(self, layer_idx, step_idx):
    # Your custom visualization code
    pass

# Add to AttentionVisualizer class
AttentionVisualizer.plot_custom_analysis = plot_custom_analysis
```

### Modifying Analysis Metrics

```python
def analyze_custom_metric(self, attention_data):
    # Your custom analysis code
    return results

# Add to AttentionAnalyzer class
AttentionAnalyzer.analyze_custom_metric = analyze_custom_metric
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_samples` or use smaller models
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Missing Data**: Ensure attention collection is enabled with `--save_attention`
4. **Visualization Errors**: Install required packages: `pip install plotly seaborn`

### Performance Tips

- Use `--max_samples 1` for quick testing
- Enable compression to reduce memory usage
- Use smaller cache sizes for faster inference
- Generate visualizations selectively to save time

## Dependencies

```
torch
transformers
matplotlib
seaborn
plotly
pandas
numpy
pickle
json
tqdm
```

## Examples

### Example 1: Basic Analysis
```bash
python run_attention_analysis.py --model llama3.1-8b-128k --compress --max_samples 1
```

### Example 2: Custom CAKE Parameters
```bash
python run_attention_analysis.py \
    --model llama3.1-8b-128k \
    --compress \
    --cache_size 256 \
    --window_size 16 \
    --tau1 1.5 \
    --tau2 0.5 \
    --gamma 100 \
    --max_samples 2
```

### Example 3: Analysis Only (Skip Collection)
```bash
python run_attention_analysis.py --skip_collection --max_samples 3
```

### Example 4: Visualization Only
```bash
python run_attention_analysis.py --skip_collection --skip_analysis --max_samples 3
```

## Contributing

To add new features:

1. Modify the appropriate component (collection, analysis, or visualization)
2. Add new metrics to the analysis utilities
3. Create new visualization functions
4. Update the README with new features
5. Test with different models and parameters

## License

This attention analysis system is part of the CAKE project and follows the same license terms.
