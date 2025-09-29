import os
import json
import torch
import numpy as np
import random
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Attention logging utilities
# -----------------------------

# attention_storage: dict[int -> list[dict]]
# Each entry: {"step": int, "stage": "prefill"|"decode", "shape": [H, S] or [H, Q, S], "tensor": np.ndarray}
attention_storage = {}
# Accumulates per-layer decode (and optionally prefill) rows to form square matrices
decode_rows_per_layer = {}  # layer_idx -> list of np.ndarray with shape [H, S_t]

def _pad_to_width(arr: np.ndarray, width: int) -> np.ndarray:
    # arr: [H, S_cur], pad right with zeros to width
    H, S = arr.shape
    if S == width:
        return arr
    out = np.zeros((H, width), dtype=arr.dtype)
    out[:, :S] = arr
    return out

def clear_attention_storage():
    attention_storage.clear()
    decode_rows_per_layer.clear()

def _ensure_layer_list(layer_idx: int):
    if layer_idx not in attention_storage:
        attention_storage[layer_idx] = []

def _to_cpu_numpy(t: torch.Tensor):
    return t.detach().to("cpu").numpy()

# def log_prefill_attentions(attentions, prefill_capture: str, ctx_len: int):
#     """
#     attentions: list of length L, each tensor [B, H, Q, S] for prefill forward
#     prefill_capture: "last" | "all" | "none"
#     ctx_len: total prompt length (tokens)
#     """
#     if prefill_capture == "none" or attentions is None:
#         return

#     # For profiling we assume B==1
#     for layer_idx, a in enumerate(attentions):
#         # a: [B, H, Q, S]
#         if a is None:
#             continue
#         a = a.squeeze(0)  # [H, Q, S]
#         if prefill_capture == "last":
#             # store only the last query row => [H, 1, S] -> squeeze to [H, S]
#             last = a[:, -1:, :]           # [H, 1, S]
#             last = last.squeeze(1)        # [H, S]
#             _ensure_layer_list(layer_idx)
#             attention_storage[layer_idx].append({
#                 "step": int(ctx_len - 1),           # the last prefill position
#                 "stage": "prefill",
#                 "shape": [int(last.shape[0]), int(last.shape[1])],
#                 "tensor": _to_cpu_numpy(last),
#             })
#         elif prefill_capture == "all":
#             # WARNING: heavy O(T^2). Store as [H, Q, S]
#             _ensure_layer_list(layer_idx)
#             attention_storage[layer_idx].append({
#                 "step": int(ctx_len - 1),           # you can mark this batch with the last index
#                 "stage": "prefill_all",
#                 "shape": [int(a.shape[0]), int(a.shape[1]), int(a.shape[2])],
#                 "tensor": _to_cpu_numpy(a),
#             })
def log_prefill_attentions(attentions, prefill_capture: str, ctx_len: int, square_include_prefill: bool):
    if (prefill_capture == "none" and not square_include_prefill) or attentions is None:
        return

    for layer_idx, a in enumerate(attentions):
        if a is None:
            continue
        a = a.squeeze(0)  # [H, Q, S] with Q == ctx_len

        # Save per your chosen prefill mode (as before)
        if prefill_capture == "last":
            last = a[:, -1, :]     # [H, S]
            _ensure_layer_list(layer_idx)
            attention_storage[layer_idx].append({
                "step": int(ctx_len - 1),
                "stage": "prefill",
                "shape": [int(last.shape[0]), int(last.shape[1])],
                "tensor": _to_cpu_numpy(last),
            })
        elif prefill_capture == "all":
            _ensure_layer_list(layer_idx)
            attention_storage[layer_idx].append({
                "step": int(ctx_len - 1),
                "stage": "prefill_all",
                "shape": [int(a.shape[0]), int(a.shape[1]), int(a.shape[2])],
                "tensor": _to_cpu_numpy(a),
            })

        # Seed the square accumulator with ALL prefill rows if requested
        if square_include_prefill:
            rows = [a[:, i, :].detach().to("cpu").numpy() for i in range(a.shape[1])]  # list of [H, S]
            decode_rows_per_layer[layer_idx] = rows


# def log_decode_attentions(attentions, step_idx: int):
#     """
#     attentions: list of length L, each tensor [B, H, 1, S] during incremental decode
#     Stores [H, S] per layer for this decode step.
#     """
#     if attentions is None:
#         return
#     for layer_idx, a in enumerate(attentions):
#         if a is None:
#             continue
#         # a: [B, H, 1, S] -> [H, S]
#         a = a.squeeze(0).squeeze(1)  # [H, S]
#         _ensure_layer_list(layer_idx)
#         attention_storage[layer_idx].append({
#             "step": int(step_idx),
#             "stage": "decode",
#             "shape": [int(a.shape[0]), int(a.shape[1])],
#             "tensor": _to_cpu_numpy(a),
#         })
def log_decode_attentions(attentions, step_idx: int, square_always: bool, current_len: int):
    """
    attentions: list[L] of [B, H, 1, S] (S == current_len)
    If square_always=True, we accumulate past rows and save a square [H, Q, S] with Q==S==current_len.
    Otherwise (legacy), we save just the current row [H, S].
    """
    if attentions is None:
        return

    for layer_idx, a in enumerate(attentions):
        if a is None:
            continue
        row = a.squeeze(0).squeeze(0)  # [H, S]

        if not square_always:
            # Legacy single-row behavior
            _ensure_layer_list(layer_idx)
            attention_storage[layer_idx].append({
                "step": int(step_idx),
                "stage": "decode",
                "shape": [int(row.shape[0]), int(row.shape[1])],
                "tensor": _to_cpu_numpy(row),
            })
            continue

        # --- Build/extend square ---
        rows = decode_rows_per_layer.get(layer_idx, [])
        # Pad old rows to new width
        rows = [_pad_to_width(r, row.shape[1]) for r in rows]
        # Append current row
        rows.append(row.detach().to("cpu").numpy())  # [H, S]

        # Stack into [Q, H, S] then transpose to [H, Q, S]
        Q = len(rows)
        square = np.stack(rows, axis=0)          # [Q, H, S]
        square = np.transpose(square, (1, 0, 2)) # [H, Q, S]

        # Save square
        _ensure_layer_list(layer_idx)
        attention_storage[layer_idx].append({
            "step": int(step_idx),
            "stage": "decode",  # keep stage label unchanged
            "shape": [int(square.shape[0]), int(square.shape[1]), int(square.shape[2])],  # [H, Q, S]
            "tensor": square,   # square with Q==S==current_len
        })

        # Keep accumulator
        decode_rows_per_layer[layer_idx] = rows

def save_attention_data(sample_idx: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"attention_sample_{sample_idx}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(attention_storage, f)
    print(f"Saved attention data for sample {sample_idx} "
          f"(layers logged: {len(attention_storage)}) -> {out_path}")

# -----------------------------
# Args & helpers
# -----------------------------

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
    parser.add_argument('--square_include_prefill', action='store_true',
                    help="Seed square matrix with ALL prefill rows so squares start from the whole prompt")
    parser.add_argument('--square_decode', action='store_true',
                        help="At each decode step, save square [H,Q,S] matrices instead of single rows")

    # New capture controls
    parser.add_argument('--prefill_capture', type=str, default="last", choices=["last", "all", "none"],
                        help="How to log attentions during prefill")
    parser.add_argument('--log_every', type=int, default=1,
                        help="Log decode attentions every N steps")
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f"<s>[INST] {prompt} [/INST]"
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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Model loading (no patching)
# -----------------------------

def load_model_eager(path, model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    # Qwen often prefers bf16; others fp16 is fine for inference
    dtype = torch.bfloat16 if "qwen2" in model_name else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="eager",  # critical for returning attentions
        device_map=None
    ).to(device)
    model.config.output_attentions = True
    model.eval()
    return model, tokenizer

# -----------------------------
# Inference with attention capture (prefill + decode)
# -----------------------------

@torch.inference_mode()
# def step_decode_and_capture(model, input_ids, max_new_tokens, log_every, prefill_capture):
#     """
#     Prefill once with output_attentions=True, log per-layer attentions as configured.
#     Then decode token-by-token, logging per-layer attentions every `log_every` steps.
#     Returns generated token ids (without the prompt).
#     """
#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device)

#     # PREFILL
#     out = model(input_ids=input_ids, use_cache=True, output_attentions=True, return_dict=True)
#     past_key_values = out.past_key_values
#     logits = out.logits[:, -1, :]

#     ctx_len = input_ids.shape[-1]
#     # attentions in prefill: list[L] of [B, H, Q, S] where Q == ctx_len
#     log_prefill_attentions(out.attentions, prefill_capture=prefill_capture, ctx_len=ctx_len)

#     # DECODE
#     generated = []
#     for t in range(max_new_tokens):
#         # greedy; switch to sampling if you want
#         next_token = torch.argmax(logits, dim=-1)  # [B]
#         generated.append(next_token)

#         out = model(input_ids=next_token.unsqueeze(-1),
#                     use_cache=True,
#                     past_key_values=past_key_values,
#                     output_attentions=True,
#                     return_dict=True)
#         past_key_values = out.past_key_values
#         logits = out.logits[:, -1, :]

#         if (t % max(1, log_every)) == 0:
#             # attentions in decode: list[L] of [B, H, 1, S]
#             # step index is absolute position in sequence
#             abs_step = ctx_len + t
#             log_decode_attentions(out.attentions, step_idx=abs_step)

#     if generated:
#         return torch.stack(generated, dim=1)  # [B, T_new]
#     return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=device)
def step_decode_and_capture(model, input_ids, max_new_tokens, log_every, prefill_capture, square_include_prefill, square_always):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # PREFILL
    out = model(input_ids=input_ids, use_cache=True, output_attentions=True, return_dict=True)
    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :]

    ctx_len = input_ids.shape[-1]
    log_prefill_attentions(out.attentions, prefill_capture=prefill_capture, ctx_len=ctx_len, square_include_prefill=square_include_prefill)

    # DECODE
    generated = []
    for t in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1)
        generated.append(next_token)

        out = model(input_ids=next_token.unsqueeze(-1),
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    return_dict=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]

        if (t % max(1, log_every)) == 0:
            abs_step = ctx_len + t
            current_len = ctx_len + t + 1  # total tokens so far (queries & keys length)
            log_decode_attentions(out.attentions, step_idx=abs_step, square_always=square_always, current_len=current_len)

    if generated:
        return torch.stack(generated, dim=1)
    return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=device)

@torch.inference_mode()
def run_inference_with_attention_capture(
    model, tokenizer, data, max_length, max_new_tokens, model_name,
    save_attention=False, max_samples=1, output_dir="./attention_data",
    prefill_capture="last", log_every=1, device=None
):
    sample_count = 0

    print(len(data))
    for json_obj in tqdm(data):
        if sample_count >= max_samples:
            break

        clear_attention_storage()

        prompt = (json_obj.get("context", "") + "\n" + json_obj.get("question", "")).strip()

        # Tokenize once to enforce max_length robustly
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
        if ids.numel() > max_length:
            # keep tail for decoder-only models
            ids = ids[-max_length:]

        # Apply chat template AFTER truncation
        if any(k in model_name for k in ["llama", "mistral", "qwen"]):
            prompt = build_chat(tokenizer, tokenizer.decode(ids, skip_special_tokens=True), model_name)

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        context_length = input_ids.shape[-1]
        print(f"Processing sample {sample_count}, input length: {context_length}")

        # Manual decode with attention capture
        # gen_ids = step_decode_and_capture(
        #     model,
        #     input_ids,
        #     max_new_tokens=max_new_tokens,
        #     log_every=log_every,
        #     prefill_capture=prefill_capture
        # )
        gen_ids = step_decode_and_capture(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            log_every=log_every,
            prefill_capture=prefill_capture,
            square_include_prefill=args.square_include_prefill,
            square_always=args.square_decode
        )

        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True) if gen_ids.numel() else ""
        
        os.makedirs("output_data", exist_ok=True)
        pred_path = os.path.join("output_data", f"output_sample_{sample_count}.json")
        full_dict = {"prompt" : prompt, "output": generated_text}
        with open(pred_path, "w") as f:
            json.dump(full_dict, f)
        print(f"Generated: {generated_text[:100]}...")

        if save_attention:
            save_attention_data(sample_count, output_dir)

        sample_count += 1

# -----------------------------
# Main
# -----------------------------

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model_name = args.model
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    # Map friendly name -> HF hub id
    model2path = {
        "llama2-7b-chat-4k": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b-chat-4k": "meta-llama/Llama-2-13b-chat-hf",
        "llama3.1-8b-128k": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistral-0.3-7b-32k": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    }
    model_path = model2path.get(model_name, model_name)

    # Load model without patching kernels
    model, tokenizer = load_model_eager(model_path, model_name, device)

    # Load test data
    print("Loading test data...")
    data_file = "data/narrativeqa.jsonl"
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()[:5]]
    else:
        raise FileNotFoundError("cannot find narrativeqa")
        data = [
            {"context": "This is a test context. " * 100, "question": "What is this about?"},
            {"context": "Another test context. " * 80, "question": "What is the main topic?"},
        ]

    # Run
    run_inference_with_attention_capture(
        model, tokenizer, data,
        args.max_length, args.max_new_tokens, model_name,
        save_attention=args.save_attention,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        prefill_capture="all",
        log_every=args.log_every,
        device=device
    )


    print("Done!")

# import os
# import json
# import torch
# import numpy as np
# import random
# import argparse
# import pickle
# from pathlib import Path
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # -----------------------------
# # Attention logging utilities
# # -----------------------------

# # attention_storage: dict[int -> list[dict]]
# # Each entry: {"step": int, "stage": "prefill"|"decode", "shape": [H, S] or [H, Q, S], "tensor": np.ndarray}
# attention_storage = {}

# def clear_attention_storage():
#     attention_storage.clear()

# def _ensure_layer_list(layer_idx: int):
#     if layer_idx not in attention_storage:
#         attention_storage[layer_idx] = []

# def _to_cpu_numpy(t: torch.Tensor):
#     return t.detach().to("cpu").numpy()

# def log_prefill_attentions(attentions, prefill_capture: str, ctx_len: int):
#     """
#     attentions: list of length L, each tensor [B, H, Q, S] for prefill forward
#     prefill_capture: "last" | "all" | "none"
#     ctx_len: total prompt length (tokens)
#     """
#     if prefill_capture == "none" or attentions is None:
#         return

#     # For profiling we assume B==1
#     for layer_idx, a in enumerate(attentions):
#         # a: [B, H, Q, S]
#         if a is None:
#             continue
#         a = a.squeeze(0)  # [H, Q, S]
#         if prefill_capture == "last":
#             # store only the last query row => [H, 1, S] -> squeeze to [H, S]
#             last = a[:, -1:, :]           # [H, 1, S]
#             last = last.squeeze(1)        # [H, S]
#             _ensure_layer_list(layer_idx)
#             attention_storage[layer_idx].append({
#                 "step": int(ctx_len - 1),           # the last prefill position
#                 "stage": "prefill",
#                 "shape": [int(last.shape[0]), int(last.shape[1])],
#                 "tensor": _to_cpu_numpy(last),
#             })
#         elif prefill_capture == "all":
#             # WARNING: heavy O(T^2). Store as [H, Q, S]
#             _ensure_layer_list(layer_idx)
#             attention_storage[layer_idx].append({
#                 "step": int(ctx_len - 1),           # you can mark this batch with the last index
#                 "stage": "prefill_all",
#                 "shape": [int(a.shape[0]), int(a.shape[1]), int(a.shape[2])],
#                 "tensor": _to_cpu_numpy(a),
#             })

# def log_decode_attentions(attentions, step_idx: int):
#     """
#     attentions: list of length L, each tensor [B, H, 1, S] during incremental decode
#     Stores [H, S] per layer for this decode step.
#     """
#     if attentions is None:
#         return
#     for layer_idx, a in enumerate(attentions):
#         if a is None:
#             continue
#         # a: [B, H, 1, S] -> [H, S]
#         a = a.squeeze(0).squeeze(1)  # [H, S]
#         _ensure_layer_list(layer_idx)
#         attention_storage[layer_idx].append({
#             "step": int(step_idx),
#             "stage": "decode",
#             "shape": [int(a.shape[0]), int(a.shape[1])],
#             "tensor": _to_cpu_numpy(a),
#         })

# def save_attention_data(sample_idx: int, output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, f"attention_sample_{sample_idx}.pkl")
#     with open(out_path, "wb") as f:
#         pickle.dump(attention_storage, f)
#     print(f"Saved attention data for sample {sample_idx} "
#           f"(layers logged: {len(attention_storage)}) -> {out_path}")

# # -----------------------------
# # Args & helpers
# # -----------------------------

# def parse_args(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, default="llama3.1-8b-128k",
#                         choices=["llama2-7b-chat-4k",
#                                  "llama2-13b-chat-4k",
#                                  "llama3.1-8b-128k",
#                                  "mistral-0.3-7b-32k",
#                                  "qwen2.5-7b-instruct"])
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--save_attention', action='store_true', help="Save attention scores")
#     parser.add_argument('--max_samples', type=int, default=1, help="Maximum number of samples")
#     parser.add_argument('--max_length', type=int, default=2048, help="Maximum input length")
#     parser.add_argument('--max_new_tokens', type=int, default=50, help="Maximum new tokens to generate")
#     parser.add_argument('--output_dir', type=str, default="./attention_data", help="Output directory")
#     # New capture controls
#     parser.add_argument('--prefill_capture', type=str, default="last", choices=["last", "all", "none"],
#                         help="How to log attentions during prefill")
#     parser.add_argument('--log_every', type=int, default=1,
#                         help="Log decode attentions every N steps")
#     return parser.parse_args(args)

# def build_chat(tokenizer, prompt, model_name):
#     if "llama3" in model_name:
#         prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#     elif "llama2" in model_name:
#         prompt = f"[INST]{prompt}[/INST]"
#     elif "mistral" in model_name:
#         prompt = f"<s>[INST] {prompt} [/INST]"
#     elif "qwen" in model_name:
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#     return prompt

# def seed_everything(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.cuda.manual_seed_all(seed)

# # -----------------------------
# # Model loading (no patching)
# # -----------------------------

# def load_model_eager(path, model_name, device):
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     # Qwen often prefers bf16; others fp16 is fine for inference
#     dtype = torch.bfloat16 if "qwen2" in model_name else torch.float16
#     model = AutoModelForCausalLM.from_pretrained(
#         path,
#         torch_dtype=dtype,
#         attn_implementation="eager",  # critical for returning attentions
#         device_map=None
#     ).to(device)
#     model.config.output_attentions = True
#     model.eval()
#     return model, tokenizer

# # -----------------------------
# # Inference with attention capture (prefill + decode)
# # -----------------------------

# @torch.inference_mode()
# def step_decode_and_capture(model, input_ids, max_new_tokens, log_every, prefill_capture):
#     """
#     Prefill once with output_attentions=True, log per-layer attentions as configured.
#     Then decode token-by-token, logging per-layer attentions every `log_every` steps.
#     Returns generated token ids (without the prompt).
#     """
#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device)

#     # PREFILL
#     out = model(input_ids=input_ids, use_cache=True, output_attentions=True, return_dict=True)
#     past_key_values = out.past_key_values
#     logits = out.logits[:, -1, :]

#     ctx_len = input_ids.shape[-1]
#     # attentions in prefill: list[L] of [B, H, Q, S] where Q == ctx_len
#     log_prefill_attentions(out.attentions, prefill_capture=prefill_capture, ctx_len=ctx_len)

#     # DECODE
#     generated = []
#     for t in range(max_new_tokens):
#         # greedy; switch to sampling if you want
#         next_token = torch.argmax(logits, dim=-1)  # [B]
#         generated.append(next_token)

#         out = model(input_ids=next_token.unsqueeze(-1),
#                     use_cache=True,
#                     past_key_values=past_key_values,
#                     output_attentions=True,
#                     return_dict=True)
#         past_key_values = out.past_key_values
#         logits = out.logits[:, -1, :]

#         if (t % max(1, log_every)) == 0:
#             # attentions in decode: list[L] of [B, H, 1, S]
#             # step index is absolute position in sequence
#             abs_step = ctx_len + t
#             log_decode_attentions(out.attentions, step_idx=abs_step)

#     if generated:
#         return torch.stack(generated, dim=1)  # [B, T_new]
#     return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=device)

# @torch.inference_mode()
# def run_inference_with_attention_capture(
#     model, tokenizer, data, max_length, max_new_tokens, model_name,
#     save_attention=False, max_samples=1, output_dir="./attention_data",
#     prefill_capture="last", log_every=1, device=None
# ):
#     sample_count = 0

#     for json_obj in tqdm(data):
#         if sample_count >= max_samples:
#             break

#         clear_attention_storage()

#         prompt = (json_obj.get("context", "") + "\n" + json_obj.get("question", "")).strip()

#         # Tokenize once to enforce max_length robustly
#         ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
#         if ids.numel() > max_length:
#             # keep tail for decoder-only models
#             ids = ids[-max_length:]

#         # Apply chat template AFTER truncation
#         if any(k in model_name for k in ["llama", "mistral", "qwen"]):
#             prompt = build_chat(tokenizer, tokenizer.decode(ids, skip_special_tokens=True), model_name)

#         enc = tokenizer(prompt, return_tensors="pt")
#         input_ids = enc.input_ids.to(device)
#         context_length = input_ids.shape[-1]
#         print(f"Processing sample {sample_count}, input length: {context_length}")

#         # Manual decode with attention capture
#         gen_ids = step_decode_and_capture(
#             model,
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             log_every=log_every,
#             prefill_capture=prefill_capture
#         )

#         generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True) if gen_ids.numel() else ""
#         print(f"Generated: {generated_text[:100]}...")

#         if save_attention:
#             save_attention_data(sample_count, output_dir)

#         sample_count += 1

# # -----------------------------
# # Main
# # -----------------------------

# if __name__ == '__main__':
#     seed_everything(42)
#     args = parse_args()
#     model_name = args.model
#     device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

#     print(f"Loading model: {model_name}")
#     print(f"Using device: {device}")

#     # Map friendly name -> HF hub id
#     model2path = {
#         "llama2-7b-chat-4k": "meta-llama/Llama-2-7b-chat-hf",
#         "llama2-13b-chat-4k": "meta-llama/Llama-2-13b-chat-hf",
#         "llama3.1-8b-128k": "meta-llama/Meta-Llama-3.1-8B-Instruct",
#         "mistral-0.3-7b-32k": "mistralai/Mistral-7B-Instruct-v0.3",
#         "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
#     }
#     model_path = model2path.get(model_name, model_name)

#     # Load model without patching kernels
#     model, tokenizer = load_model_eager(model_path, model_name, device)

#     # Load test data
#     print("Loading test data...")
#     data_file = "data/narrativeqa.jsonl"
#     if os.path.exists(data_file):
#         with open(data_file, 'r', encoding='utf-8') as f:
#             data = [json.loads(line) for line in f.readlines()[:5]]
#     else:
#         data = [
#             {"context": "This is a test context. " * 100, "question": "What is this about?"},
#             {"context": "Another test context. " * 80, "question": "What is the main topic?"},
#         ]

#     # Run
#     run_inference_with_attention_capture(
#         model, tokenizer, data,
#         args.max_length, args.max_new_tokens, model_name,
#         save_attention=args.save_attention,
#         max_samples=args.max_samples,
#         output_dir=args.output_dir,
#         prefill_capture=args.prefill_capture,
#         log_every=args.log_every,
#         device=device
#     )

#     print("Done!")
