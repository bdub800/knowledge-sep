import os

import torch
from tqdm import tqdm
import argparse
import json
import pandas as pd

from model import instantiate_model
from data import get_generation_dataloader
from utils import sample_tokens, process_answer

torch.serialization.add_safe_globals([argparse.Namespace])

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

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for processing text
        eval_loader: DataLoader that yields tokenized prompts and ground truth answers
        device: Device to run on
        config: Configuration object with sampling parameters and where to save prompts and generations etc.

    Returns:
        Dictionary with accuracy and other metrics, generation data
    """
    model.eval()

    correct = 0
    total = 0
    accuracy = 0
    eval_data = []
    new_tokens = 0
    num_batches = 0

    progress_bar = tqdm(eval_loader, desc="Generating & Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truths = batch['ground_truths']

            prompt_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

            batch_size = input_ids.shape[0]

            # Generate tokens autoregressively for the whole batch
            generated_ids = input_ids.clone()

            # Track which sequences have finished (generated EOS)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Initial base model forward pass on full prompt with KV caching
            base_outputs = model.base_model.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
            original_input = base_outputs.last_hidden_state
            past_key_values = base_outputs.past_key_values

            for i in range(config.max_new_tokens):
                # Get init states for current sequence length
                output_states, latent_states = model.get_inits(generated_ids)
                if getattr(config, 'verbose', False) and (i % 100 == 0):
                    print(f'output states shape {output_states.shape}; latent states shape {latent_states.shape}')

                for sup_step in range(config.N_supervision):
                    output_states, latent_states, logits = model.deep_recursion(
                        original_input=original_input,
                        output_states=output_states,
                        latent_states=latent_states,
                        attention_mask=attention_mask,
                        n=config.n_latent_recursions,
                        T=config.T_outer_loops,
                    )

                # Sample the next token
                next_token_logits = logits[:, -1, :]
                next_tokens = sample_tokens(next_token_logits, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)

                # Append the new tokens
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                
                new_tokens += 1

                if getattr(config, 'verbose', False):
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                    print(f'Generating {i}-th new token ...')
                    print('GEN TEXTS ' + '>'*40)
                    print(generated_texts[0])
                    print('<'*50)
                else:
                    progress_bar.set_postfix({
                        'accuracy': f'{accuracy:.4f}',
                        'correct': correct,
                        'total': total,
                        'new_tokens/sample': new_tokens,
                    })
                
                # Mark sequences that generated EOS
                finished |= (next_tokens.squeeze(-1) == tokenizer.eos_token_id)

                # Stop if all sequences have finished
                if finished.all():
                    break

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)

                # Incremental base model forward: only process the new token
                base_outputs = model.base_model.model(
                    input_ids=next_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                if getattr(config, 'verbose', False) and (i % 100 == 0):
                    print(f'original_input shape {original_input.shape}; last_hidden_state shape {base_outputs.last_hidden_state.shape}')
                original_input = torch.cat([original_input, base_outputs.last_hidden_state], dim=1)
                past_key_values = base_outputs.past_key_values

            # Batch decode all sequences at once
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

            # Evaluate each sequence in the batch
            for i in range(batch_size):
                # Extract the answer (remove the prompt)
                if generated_texts[i].startswith(prompt_texts[i]):
                    generated_answer = generated_texts[i][len(prompt_texts[i]):].strip()
                else:
                    generated_answer = generated_texts[i].strip()

                final_answer, thinking = process_answer(tokenizer, generated_answer)

                try:
                    is_match = float(final_answer) == float(ground_truths[i])
                except ValueError:
                    is_match = (final_answer.lower() == ground_truths[i].lower())
                correct += int(is_match)
                total += 1

                eval_data.append({
                    'id': total,
                    'prompt': prompt_texts[i],
                    'generated_answer': generated_answer,
                    'thinking': thinking,
                    'final_answer': final_answer,
                    'is_match': is_match,
                    'ground_truth': ground_truths[i]
                })

                if hasattr(config, "save_eval_interval") and config.save_eval_interval > 0:
                    if total % config.save_eval_interval == 0:
                        pd.DataFrame(eval_data).to_json(config.save_eval_data_path, lines=True, orient='records')

            # Update progress bar after each batch
            accuracy = correct / total if total > 0 else 0
            num_batches += 1
            progress_bar.set_postfix({
                'accuracy': f'{accuracy:.4f}',
                'correct': correct,
                'total': total,
                'new_tokens/sample': new_tokens / num_batches,
            })

    eval_dict = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'new_tokens/sample': new_tokens / num_batches,
    }
    return eval_dict, eval_data

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on some dataset')

    # Model arguments
    parser.add_argument('--ckpt_path', type=str, help='Path to saved model')

    # Evaluation arguments
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Evaluation batch size')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Max new tokens to generate during generation eval')
    parser.add_argument('--max_length', type=int, default=32768,
                        help='Maximum sequence length')
    parser.add_argument('--N_supervision', type=int, default=1,
                        help='Number of deep supervision steps')

    # Dataset arguments
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of evaluation samples (None for all)')
    parser.add_argument('--train_set', action='store_true',
                        help='Eval on train set instead of test set')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_eval_interval', type=int, default=100,
                        help='Save the evalution data every x examples')
    parser.add_argument('--save_eval_dir', type=str, default=None,
                        help='Dir to save eval data')
    parser.add_argument('--verbose', action='store_true',
                        help='Print generated text at each token step')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint and rebuild model
    print(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    train_config = checkpoint['config']

    # Use training config for model hyperparams if not overridden
    args.base_model = train_config.base_model
    args.num_recurrent_layers = train_config.num_recurrent_layers
    # args.N_supervision = train_config.N_supervision
    args.n_latent_recursions = train_config.n_latent_recursions
    args.T_outer_loops = train_config.T_outer_loops

    tokenizer, model = instantiate_model(args.base_model, args.num_recurrent_layers, device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully.")
    
    eval_loader = get_generation_dataloader(
        tokenizer, args.max_length, args.eval_batch_size,
        seed=args.seed, train=args.train_set, num_samples=args.num_eval_samples
    )

    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")

    # Create save directory
    os.makedirs(args.save_eval_dir, exist_ok=True)

    eval_dict_name = args.ckpt_path.split('/')[-1] + '.dict'
    eval_data_name = args.ckpt_path.split('/')[-1] + '.jsonl'
    eval_dict_path = os.path.join(args.save_eval_dir, eval_dict_name)
    eval_data_path = os.path.join(args.save_eval_dir, eval_data_name)

    args.save_eval_data_path = eval_data_path
    
    eval_dict, eval_data = evaluate_generation(model, tokenizer, eval_loader, device, args)
    
    with open(eval_dict_path, 'w') as f:
        json.dump(eval_dict, f, indent=4)
    pd.DataFrame(eval_data).to_json(eval_data_path, lines=True, orient='records')

    print("\n" + "="*50)
    print("Evalution complete!")
    print(f"Eval dict: {eval_dict}")
    print("="*50)


if __name__ == "__main__":
    main()