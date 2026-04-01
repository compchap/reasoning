from reasoning_from_scratch.qwen3 import KVCache
import torch
# import extract_final_candidate from model_eval_utils 

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

# modified generate_text_stream_concat function, to pass function for text generation
def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False, 
    # parameters to accept a text generation function and additional arguments
    generate_func=None,
    **generate_kwargs
):
    # If the text generation function is undefined, we use generate_text_basic_stream_cache
    if generate_func is None:
        generate_func = generate_text_basic_stream_cache
        
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)
 
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        # We can pass additional arguments to the text generation function if needed
        **generate_kwargs,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
 
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)

def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature
  
@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Get logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
 
        ########################################
        # NEW:
        orig_device = token_ids.device
 
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        else: 
            logits = scale_logits_by_temperature(out, temperature) # Apply temperature scaling on logits
            probas = torch.softmax(logits, dim=-1) # Convert to probabilities
            next_token = torch.multinomial(probas.cpu(), num_samples=1) # Sample token according to probabilities
            next_token = next_token.to(orig_device)
            
        #########################################
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break
 
        yield next_token
        out = model(next_token, cache=cache)[:, -1]

def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    # Sort by descending probability
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    # Cumulative sum
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Keep tokens where prefix cumulative mass (before each token) is < top_p
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    # For top_p <= 0, only the highest-probability token is guaranteed to be kept as a fallback
    keep[:, 0] = True

    # Zero out beyond cutoff
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )

    # Map back to original order
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
 
    # Renormalize to sum to 1
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom

@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.,
    top_p=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
 
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
 
        orig_device = token_ids.device
 
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
 
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
 
            probas = top_p_filter(probas, top_p)
 
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)
 
        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break
 
        yield next_token
        out = model(next_token, cache=cache)[:, -1]

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

# Self Consistency Vote
from collections import Counter
 
def self_consistency_vote(
    model, 
    tokenizer, 
    prompt, 
    device,
    num_samples=10, 
    temperature=0.8, 
    top_p=0.9, 
    max_new_tokens=2048,
    show_progress=True, 
    show_long_answer=False, 
    seed=None,
):
    full_answers, short_answers = [], []
 
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)
 
        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, 
            device=device,
            max_new_tokens=max_new_tokens, 
            verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, 
            top_p=top_p,
        )
 
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")
 
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)
 
    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None
 
    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
