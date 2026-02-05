import torch
from tqdm import tqdm

def evaluate(model, eval_loader, device, config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(eval_loader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output_states, latent_states = model.get_inits(input_ids)

            loss = None
            for sup_step in range(config.N_supervision):
                # Forward pass
                output_states, latent_states, _, loss = model(
                    output_states=output_states,
                    latent_states=latent_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    n=config.n_latent_recursions,
                    T=config.T_outer_loops,
                )

            # Only record loss from the last supervision step
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({'loss': total_loss / num_batches})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_generation(model, tokenizer, eval_loader, device, config):
    """
    Evaluate the model by generating answers from questions and comparing to ground truth.
    Uses batch generation for efficiency.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing text
        eval_loader: DataLoader that yields tokenized prompts and ground truth answers
        device: Device to run on
        config: Configuration object with model parameters
        max_new_tokens: Maximum number of tokens to generate
        num_samples: Number of samples to evaluate (None for all)

    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()

    correct = 0
    total = 0

    progress_bar = tqdm(eval_loader, desc="Generating & Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truths = batch['ground_truths']

            batch_size = input_ids.shape[0]

            # Generate tokens autoregressively for the whole batch
            generated_ids = input_ids.clone()

            # Track which sequences have finished (generated EOS)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Init kv cache for new batch
            past_key_values = None

            for i in range(config.max_new_tokens):
                print (f'Generating {i}-th new token ...')
                
                # Get base model embeddings
                base_outputs = model.base_model.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                original_input = base_outputs.last_hidden_state
                past_key_values = base_outputs.past_key_values

                # Get init states potentially with new tokens appended to context
                output_states, latent_states = model.get_inits(generated_ids)

                for sup_step in range(config.N_supervision):
                    output_states, latent_states, logits, _ = model.deep_recursion(
                        original_input=original_input,
                        output_states=output_states,
                        latent_states=latent_states,
                        attention_mask=attention_mask,
                        labels=None,  # No labels needed for generation
                        n=config.n_latent_recursions,
                        T=config.T_outer_loops,
                    )

                # Get the next token from logits (greedy decoding)
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Mark sequences that generated EOS
                finished |= (next_tokens.squeeze(-1) == tokenizer.eos_token_id)

                # Stop if all sequences have finished
                if finished.all():
                    break

                # Append the new tokens
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                # for debug
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                print('GEN TEXTS ' + '>'*40)
                print(generated_texts[0])
                print('<'*50)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)

            # Batch decode all sequences at once
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            prompt_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            # Evaluate each sequence in the batch
            for i in range(batch_size):
                # Extract the answer (remove the prompt)
                if generated_texts[i].startswith(prompt_texts[i]):
                    generated_answer = generated_texts[i][len(prompt_texts[i]):].strip()
                else:
                    generated_answer = generated_texts[i].strip()

                # Check if the generated answer matches the ground truth (exact string match)
                if generated_answer.lower() == ground_truths[i].lower():
                    correct += 1

                total += 1

            # Update progress bar
            accuracy = correct / total if total > 0 else 0
            progress_bar.set_postfix({
                'accuracy': f'{accuracy:.4f}',
                'correct': correct,
                'total': total
            })

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }