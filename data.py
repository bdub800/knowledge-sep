from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding


def turn_to_standard_convo(example):
    thinking, ans = example['answer'].split('####')
    thinking = thinking.strip()
    ans = ans.strip()
    standard_convo = [
        {'role': 'user', 'content': example['question']},
        {'role': 'assistant', 'content': f'<think>\n{thinking}\n</think>\nAnswer: {ans}'}
    ]
    return standard_convo

def tokenize(example, tokenizer, max_length):
    text = tokenizer.apply_chat_template(
        turn_to_standard_convo(example),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    return tokenizer(text, max_length=max_length, truncation=True)

def get_dataloader(tokenizer, max_length, batch_size, seed, train=True):
    # Load dataset
    print(f"Loading dataset...")

    # Load in streaming mode
    ds = load_dataset(
        "openai/gsm8k", "main",
        split='train' if train else 'test',
        streaming=True,
    )
    if train:
        ds = ds.shuffle(buffer_size=10_000, seed=seed)

    ds = ds.map(lambda row: tokenize(row, tokenizer, max_length), remove_columns=ds.column_names)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator
    )


def get_generation_dataloader(tokenizer, max_length, batch_size, seed, train=False):
    """
    Dataloader for generation evaluation that only provides the question part.
    Returns tokenized prompts with ground truth answers.
    Pads on the LEFT for batch generation (so all sequences end at the same position).
    """
    print(f"Loading dataset for generation evaluation...")

    # Load dataset
    ds = load_dataset(
        "openai/gsm8k", "main",
        split='train' if train else 'test',
        streaming=True,
    )
    if train:
        ds = ds.shuffle(buffer_size=10_000, seed=seed)

    # Extract question and ground truth answer
    def prepare_for_generation(example):
        # Extract ground truth answer (after ####)
        parts = example['answer'].split('####')
        if len(parts) == 2:
            ground_truth = parts[1].strip()
        else:
            ground_truth = ""

        # Format as chat and apply template
        messages = [
            {'role': 'user', 'content': example['question']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        return {
            'prompt': prompt,
            'ground_truth': ground_truth
        }

    ds = ds.map(prepare_for_generation, remove_columns=ds.column_names)

    # Custom collate function for LEFT padding (important for generation)
    def collate_fn(batch):
        prompts = [item['prompt'] for item in batch]
        ground_truths = [item['ground_truth'] for item in batch]

        # Tokenize with LEFT padding for batch generation
        tokenized = tokenizer(
            prompts,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
            padding_side='left'  # Left padding for generation
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'ground_truths': ground_truths
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn
    )