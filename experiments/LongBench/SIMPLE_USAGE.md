# Simple CAKE Attention Analysis

This is a simplified implementation for collecting attention scores from CAKE models at each layer for each inference step.

## Files

1. **`pred_cake_simple_attention.py`** - Main script to collect attention data
2. **`modify_llama_simple.py`** - Modified Llama attention to capture attention weights
3. **`simple_visualize.py`** - Simple visualization of attention patterns

## Quick Usage

### 1. Collect Attention Data

```bash
# Run with CAKE compression enabled
python pred_cake_simple_attention.py \
    --model llama3.1-8b-128k \
    --compress \
    --save_attention \
    --max_samples 1 \
    --dataset narrativeqa

# Run without compression
python pred_cake_simple_attention.py \
    --model llama3.1-8b-128k \
    --save_attention \
    --max_samples 1 \
    --dataset narrativeqa
```

### 2. Visualize Attention Data

```bash
python simple_visualize.py
```

## What Gets Captured

- **Attention weights** at each layer for each inference step
- **Shape information** for debugging
- **Compressed vs uncompressed** attention patterns
- **Layer-wise attention** differences

## Output Structure

```
attention_data/
└── attention_sample_0.pkl    # Contains attention data for the first sample
```

The pickle file contains:
```python
{
    0: {  # Step 0 (prefill)
        0: {'shape': [batch, heads, seq, seq], 'tensor': numpy_array},  # Layer 0
        1: {'shape': [batch, heads, seq, seq], 'tensor': numpy_array},  # Layer 1
        ...
    },
    1: {  # Step 1 (first generated token)
        0: {'shape': [batch, heads, seq, seq], 'tensor': numpy_array},  # Layer 0
        1: {'shape': [batch, heads, seq, seq], 'tensor': numpy_array},  # Layer 1
        ...
    },
    ...
}
```

## Key Features

- **Minimal complexity**: Only captures what's needed
- **Memory efficient**: Stores on CPU, converts to numpy
- **Layer-by-layer**: Separate attention weights per layer
- **Step tracking**: Tracks inference steps automatically
- **Easy visualization**: Simple plotting functions

## Customization

### Add More Models

Edit `pred_cake_simple_attention.py` in the `load_model_and_tokenizer` function:

```python
elif "your_model" in model_name:
    # Add your model's attention modification
    pass
```

### Modify What Gets Captured

Edit `modify_llama_simple.py` in the `llama_attn_forward_with_storage` function:

```python
# Store additional information
store_attention(self.layer_idx, inference_step, additional_data)
```

### Custom Visualizations

Edit `simple_visualize.py`:

```python
def your_custom_plot(attention_data):
    # Your visualization code
    pass
```

## Troubleshooting

**Out of Memory**: Reduce `max_samples` or use smaller models
**No data saved**: Make sure `--save_attention` flag is used
**Import errors**: Run from the correct directory
**CUDA errors**: Use `--device 0` or reduce batch size
