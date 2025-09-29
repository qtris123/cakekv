# Basic Llama Attention Analysis

A simple implementation to capture attention weights from normal Llama forward pass.

## Files

1. **`simple_llama_attention.py`** - Main script to collect attention data
2. **`simple_view_attention.py`** - Simple visualization of attention patterns

## Quick Usage

### 1. Collect Attention Data

```bash
# Basic usage - collect attention for 1 sample
python simple_llama_attention.py --model llama3.1-8b-128k --save_attention

# Multiple samples
python simple_llama_attention.py --model llama3.1-8b-128k --save_attention --max_samples 3

# Custom settings
python simple_llama_attention.py \
    --model llama3.1-8b-128k \
    --save_attention \
    --max_samples 2 \
    --max_length 1024 \
    --max_new_tokens 20
```

### 2. View Attention Data

```bash
python simple_view_attention.py
```

## What Gets Captured

- **Attention weights** from all layers during generation
- **Layer-wise attention** patterns  
- **Head-wise attention** patterns
- **Shape information** for debugging

## Output Structure

```
attention_data/
└── attention_sample_0.pkl    # Contains attention data
```

The pickle file contains:
```python
{
    0: {  # Layer 0
        'shape': [batch_size, num_heads, seq_len, seq_len], 
        'tensor': numpy_array
    },
    1: {  # Layer 1
        'shape': [batch_size, num_heads, seq_len, seq_len], 
        'tensor': numpy_array
    },
    ...
}
```

## Visualizations

### 1. Attention Heatmaps
- Shows attention weights between query and key positions
- Color intensity represents attention strength
- Can plot specific layer and head

### 2. Layer Comparisons
- Side-by-side comparison across layers
- Shows how attention patterns differ between layers

### 3. Head Comparisons  
- Compare different attention heads in the same layer
- Reveals specialized head behaviors

## Command Line Options

```bash
python simple_llama_attention.py \
    --model MODEL_NAME \              # Model to use
    --device DEVICE_ID \              # GPU device (default: 0)
    --save_attention \                # Enable saving attention data
    --max_samples N \                 # Number of samples to process
    --max_length LENGTH \             # Maximum input sequence length
    --max_new_tokens TOKENS \         # Maximum tokens to generate
    --output_dir DIR                  # Output directory
```

## Supported Models

- `llama2-7b-chat-4k`
- `llama2-13b-chat-4k` 
- `llama3.1-8b-128k`
- `mistral-0.3-7b-32k`
- `qwen2.5-7b-instruct`

## Example Output

```
Processing sample 0, input length: 1247
Generated: This is about artificial intelligence and machine learning...

Saved attention data for sample 0 with 32 layers
Done!
```

## Memory Considerations

- Attention weights are stored on CPU to reduce GPU memory usage
- Large sequences may require reducing max_length
- Multiple samples increase memory usage proportionally

## Troubleshooting

**Out of Memory**: 
- Reduce `max_length` or `max_new_tokens`
- Use smaller models
- Reduce `max_samples`

**Slow Generation**:
- This uses eager attention (slower but captures weights)
- Consider reducing sequence length

**Import Errors**:
- Make sure transformers is installed: `pip install transformers`
- Run from the correct directory

## Key Features

- **Simple**: No CAKE complexity, just standard Llama
- **Clean**: Minimal code, easy to understand
- **Complete**: Captures all layer attention weights
- **Visual**: Easy plotting and analysis
