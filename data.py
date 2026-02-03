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