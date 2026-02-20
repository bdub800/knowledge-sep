import re

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def turn_to_standard_convo(example, enable_cot):
    thinking, ans = example['answer'].split('####')
    thinking = thinking.strip()
    ans = ans.strip()
    # get rid of things like <<200/2=100>> in GSM8K
    thinking = re.sub(r'<<.*?>>', '', thinking)
    if enable_cot == 'standard':
        content = f'<think>\n{thinking}\n</think>\n\n**Final Answer:** $\\boxed{{{ans}}}$'
    elif enable_cot == 'after_scratch_pad':
        content = f'<think>\n\n</think>\n\n{thinking}\n\n**Final Answer:** $\\boxed{{{ans}}}$'
    elif enable_cot == 'none_at_all':
        content = f'<think>\n\n</think>\n\n**Final Answer:** $\\boxed{{{ans}}}$'
    else:
        raise ValueError(f'enable_cot got value {enable_cot} which is not in the list of options')

    standard_convo = [
        {'role': 'user', 'content': example['question']},
        {'role': 'assistant', 'content': content}
    ]
    return standard_convo

def prepare_text(example, tokenizer, enable_cot):
    whole_text = tokenizer.apply_chat_template(
        turn_to_standard_convo(example, enable_cot),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True
    )
    just_question = [{'role': 'user', 'content': example['question']}]
    prompt_text = tokenizer.apply_chat_template(
        just_question,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return {'whole_text': whole_text, 'prompt_text':prompt_text}
    
def get_dataloader(tokenizer, max_length, batch_size, seed, train=True, num_samples=None, enable_cot='standard'):
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
    if num_samples:
        ds = ds.take(num_samples)

    ds = ds.map(lambda row: prepare_text(row, tokenizer, enable_cot=enable_cot), remove_columns=ds.column_names)

#     collator = DataCollatorWithPadding(
#         tokenizer=tokenizer,
#         return_tensors="pt"
#     )

    def collate_fn(batch):
        wholes = [item['whole_text'] for item in batch]
        prompts = [item['prompt_text'] for item in batch]

        tokenized = tokenizer(
            wholes,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
            padding_side='right'
        )
        tokenized_prompts = tokenizer(
            prompts,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
            padding_side='right'
        )
        
        attn_mask_whole = tokenized['attention_mask']
        lengths = tokenized_prompts['attention_mask'].sum(dim=-1)
        # Mask out the loss on prompt tokens, prompt tokens should not be prediction targets
        col_indices = torch.arange(attn_mask_whole.size(1)).unsqueeze(0)
        mask = col_indices < lengths.unsqueeze(1) # no need for lengths - 1 here because loss mask shifted later in compute loss
        loss_mask = attn_mask_whole.clone()
        loss_mask[mask] = 0

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'loss_mask': loss_mask
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn
    )


def get_generation_dataloader(tokenizer, max_length, batch_size, seed, train=False, num_samples=None, enable_cot='standard'):
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
    if num_samples:
        ds = ds.take(num_samples)

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
        if enable_cot == 'standard':
            enable_thinking = True
        elif enable_cot in ['after_scratch_pad', 'none_at_all']:
            enable_thinking = False
        else:
            raise ValueError(f'enable_cot got value {enable_cot} which is not in the list of options')
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
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
            padding_side='left'
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