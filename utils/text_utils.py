from reasoning_from_scratch.qwen3 import KVCache
import torch

def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
        
        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
 
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
 
    else:
        device = torch.device("cpu")
        print("Using CPU")
 
    return device

# Disable gradient tracking for speed and memory efficiency
@torch.inference_mode()
def generate_text_basic(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    # Switch model to evaluation mode to enable deterministic behavior (best practice)
    model.eval()
 
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break

        # concatenating new token at the end of prev. tokens
        token_ids = torch.cat(
            [token_ids, next_token], dim=1)
    
    # Slicing off the original prompt    
    return token_ids[:, input_length:]

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    # input_length = token_ids.shape[1]
    model.eval()

    # input_length = token_ids.shape[1]
    
    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break
            
        # We now yield each token as it's generated
        yield next_token
 
        token_ids = torch.cat(
            [token_ids, next_token], dim=1)

        # Since we use yield, we no longer need the return statement
        # return token_ids[:, input_length:]
        

# implement KV caching
@torch.inference_mode()
def generate_text_basic_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
 
    input_length = token_ids.shape[1]
    model.eval()
    # initialize KV cache
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    # initially full input is provided to the model
    out = model(token_ids, cache=cache)[:, -1]
 
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break
 
        token_ids = torch.cat([token_ids, next_token], dim=1)
        # subsequent iterations only feel next_token to the model
        out = model(next_token, cache=cache)[:, -1]
 
    return token_ids[:, input_length:]

# Text Generation using base model with Stream and KV caching enabled
@torch.inference_mode()
def generate_text_basic_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    # Input length is no longer needed
    # input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
 
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break
 
        yield next_token
        # token_ids = torch.cat([token_ids, next_token], dim=1)
        out = model(next_token, cache=cache)[:, -1]
 
    # return token_ids[:, input_length:]

# Wrapper for Streamed Text Generation
def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    # Encode prompt text into token IDs and place on device
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device  
        ).unsqueeze(0)                           
 
    generated_ids = []
    # Stream tokens one by one using cached generation
    for token in generate_text_basic_stream_cache(
        model=model,                                
        token_ids=input_ids,                        
        max_new_tokens=max_new_tokens,              
        eos_token_id=tokenizer.eos_token_id,        
    ):                                              
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
 
        # Optionally print tokens as they are generated
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    # Decode all generated IDs into final text string
    return tokenizer.decode(generated_ids)

# Generate Stats
def generate_stats(output_token_ids, tokenizer, start_time,
                   end_time, print_tokens=True):
    total_time = end_time - start_time
    print(f"Time: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
 
    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():
            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
 
    if print_tokens:
        output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())
        print(f"\n{output_text}")
