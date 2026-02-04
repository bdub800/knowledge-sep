import os
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

from model import ModelWithRecurrentHead, Qwen3RecurrentModule
from data import get_dataloader


def train_epoch(model, train_loader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        output_states, latent_states = model.get_inits(input_ids)

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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                f'loss_{sup_step}': loss.item(),
                'avg_loss': total_loss / num_batches,
                'lr': scheduler.get_last_lr()[0]
            })

    return total_loss / num_batches


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

            # Forward pass
            output_states, latent_states, _, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                n=config.n_latent_recursions,
                T=config.T_outer_loops,
            )

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': total_loss / num_batches})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train ModelWithRecurrentHead on some dataset')

    # Model arguments
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-0.6B-Base',
                        help='Base model name or path')
    parser.add_argument('--num_recurrent_layers', type=int, default=2,
                        help='Number of layers in recurrent module')
    parser.add_argument('--n_latent_recursions', type=int, default=6,
                        help='Number of latent recursions (n parameter)')
    parser.add_argument('--T_outer_loops', type=int, default=3,
                        help='Number of outer loops (T parameter)')
    parser.add_argument('--N_supervision', type=int, default=8,
                        help='Number of deep supervision steps')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--max_length', type=int, default=32768,
                        help='Maximum sequence length')

    # Dataset arguments
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Number of training samples (None for all)')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of evaluation samples (None for all)')

    # Logging and saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_steps', type=int, default=None,
                        help='Save checkpoint every N steps (None to save only at end of epoch)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and base model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False # we don't want to have optimizer states for these

    print(f"Base model frozen. Trainable parameters: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")

    # Create recurrent head config
    new_config = copy.deepcopy(base_model.config)
    new_config.num_hidden_layers = args.num_recurrent_layers
    print(f"Recurrent head config: {args.num_recurrent_layers} layers")

    # Create recurrent head
    custom_head = Qwen3RecurrentModule(new_config).to(device).to(torch.bfloat16)

    # Create complete model
    model = ModelWithRecurrentHead(base_model, custom_head).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. Trainable parameters: {trainable_params:,} / {total_params:,}")

    train_loader = get_dataloader(tokenizer, args.max_length, args.batch_size, seed=args.seed, train=True)
    eval_loader = get_dataloader(tokenizer, args.max_length, args.eval_batch_size, seed=args.seed, train=False)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    num_training_steps = len(train_loader) * args.N_supervision * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    best_eval_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        eval_loss = evaluate(model, eval_loader, device, args)
        print(f"Eval loss: {eval_loss:.4f}")

        # Save checkpoint if best
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'config': args,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # Save epoch checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'config': args,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
