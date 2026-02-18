import torch
import torch.nn as nn

def compute_shift_lm_loss(logits, labels, vocab_size, loss_mask=None):
    """
    Compute cross-entropy loss for language modeling.

    Args:
        loss_mask: Optional mask of shape (batch, seq_len). 0 = exclude from loss, 1 = include.
    """
    logits = logits.to(torch.float32) # upcast for loss calculation

    # Labels is actually passed in as labels=input_ids
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct( # per token loss
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    ).view(shift_labels.shape)

    if loss_mask is not None:
        # Shift the mask to align with shifted logits/labels
        shift_mask = loss_mask[..., 1:].contiguous()
        loss = loss * shift_mask
        loss = loss.sum() / shift_mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    return loss