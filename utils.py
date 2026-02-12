import math
import torch

# From https://github.com/SamsungSAILMontreal/TinyRecursiveModels
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def sample_tokens(logits, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0):
    """Sample from logits with temperature, min-p, top-k, and top-p filtering.

    Args:
        logits: (batch_size, vocab_size) raw logits for the next token.
        temperature: Scaling factor applied before softmax.
        top_p: Nucleus sampling threshold.
        top_k: Keep only top-k logits before sampling.
        min_p: Minimum probability relative to the max probability token.

    Returns:
        (batch_size, 1) sampled token ids.
    """
    # 1. Temperature
    if temperature == 0: # greedy
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # 2. Min-p filtering
    if min_p > 0.0:
        max_prob = probs.max(dim=-1, keepdim=True).values
        probs = probs.masked_fill(probs < min_p * max_prob, 0.0)
        # Re-normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # 3. Top-k filtering
    if top_k > 0:
        top_k_values, _ = torch.topk(probs, top_k, dim=-1)
        min_top_k = top_k_values[:, -1, None]
        probs = probs.masked_fill(probs < min_top_k, 0.0)
        # Re-normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # 4. Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs > top_p # everything in the past, not incl. current pos, is >top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        # Re-normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens