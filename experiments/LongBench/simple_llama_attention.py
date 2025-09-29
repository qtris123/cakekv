import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import pickle
from pathlib import Path

# Global storage for attention scores
attention_storage = {}

def store_attention(layer_idx, attention_weights):
    """Store attention weights for a specific layer"""
    if attention_weights is not None:
        # Store attention weights on CPU to save GPU memory
        attention_storage[layer_idx] = attention_weights.detach().cpu()

def clear_attention_storage():
    """Clear the attention storage"""
    global attention_storage
    attention_storage = {}

def save_attention_data(sample_idx, output_dir):
    """Save attention data to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy for saving
    data_to_save = {}
    for layer_idx, attention_weights in attention_storage.items():
        data_to_save[layer_idx] = {
            'shape': list(attention_weights.shape),
            'tensor': attention_weights.numpy()
        }
    
    # Save as pickle
    with open(f"{output_dir}/attention_sample_{sample_idx}.pkl", 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"Saved attention data for sample {sample_idx} with {len(data_to_save)} layers")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k", 
                        choices=["llama2-7b-chat-4k", 
                                 "llama2-13b-chat-4k", 
                                 "llama3.1-8b-128k", 
                                 "mistral-0.3-7b-32k",
                                 "qwen2.5-7b-instruct"])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_attention', action='store_true', help="Save attention scores")
    parser.add_argument('--max_samples', type=int, default=1, help="Maximum number of samples")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum input length")
    parser.add_argument('--max_new_tokens', type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument('--output_dir', type=str, default="./attention_data", help="Output directory")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    """Build chat prompt format"""
    if "llama3" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt

@torch.inference_mode()
def run_inference_with_attention_capture(model, tokenizer, data, max_length, max_new_tokens, model_name, save_attention=False, max_samples=1, output_dir="./attention_data"):
    
    sample_count = 0
    
    for json_obj in tqdm(data):
        if sample_count >= max_samples:
            break
            
        # Clear attention storage for this sample
        clear_attention_storage()
        
        prompt = json_obj.get("context", "") + "\n" + json_obj.get("question", "")
        
        # Truncate to fit max_length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # Build chat format
        if "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
            prompt = build_chat(tokenizer, prompt, model_name)

        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_ids.input_ids.shape[-1]

        print(f"Processing sample {sample_count}, input length: {context_length}")
        
        # Generate with attention capture
        output = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            output_attentions=True,  # This enables attention output
        )

        # Get the generated text
        generated_ids = output[0][context_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Generated: {generated_text[:100]}...")
        
        # Save attention data if requested
        if save_attention:
            save_attention_data(sample_count, output_dir)
        
        sample_count += 1

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_with_attention_modification(path, model_name, device):
    """Load model and modify attention to capture weights"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # Load model
    dtype = torch.float16 if "qwen2" not in model_name else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=dtype,
        attn_implementation="eager",  # Use eager attention to get weights
    ).to(device)
    
    # Modify attention forward to capture weights
    def attention_forward_with_capture(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=True, use_cache=False, **kwargs):
        """Modified attention forward that captures attention weights"""
        
        # Standard attention computation
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Rotary positional embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads if necessary
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # STORE ATTENTION WEIGHTS
        store_attention(getattr(self, 'layer_idx', 0), attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

    # Replace attention forward for all layers
    for i, layer in enumerate(model.model.layers):
        # Add layer index attribute
        layer.self_attn.layer_idx = i
        # Replace the forward method
        layer.self_attn.forward = attention_forward_with_capture.__get__(layer.self_attn, layer.self_attn.__class__)
    
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model_name = args.model
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    # Model paths
    model2path = {
        "llama2-7b-chat-4k": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b-chat-4k": "meta-llama/Llama-2-13b-chat-hf", 
        "llama3.1-8b-128k": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral-0.3-7b-32k": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct"
    }
    
    model_path = model2path.get(model_name, model_name)
    
    # Load model with attention modification
    model, tokenizer = load_model_with_attention_modification(model_path, model_name, device)

    # Load test data
    print("Loading test data...")
    data_file = "data/narrativeqa.jsonl"
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()[:5]]  # Load first 5 samples
    else:
        # Create dummy data for testing
        data = [
            {"context": "This is a test context. " * 100, "question": "What is this about?"},
            {"context": "Another test context. " * 80, "question": "What is the main topic?"},
        ]
    
    # Run inference
    run_inference_with_attention_capture(
        model, tokenizer, data, 
        args.max_length, args.max_new_tokens, model_name,
        save_attention=args.save_attention, 
        max_samples=args.max_samples, 
        output_dir=args.output_dir
    )
    
    print("Done!")
