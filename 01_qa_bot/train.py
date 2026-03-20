"""
Project 01: Domain-Specific Q&A Bot
Fine-tunes DistilBERT on a Q&A dataset for FAQ-style support workflows.
"""

import json
import os
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
import torch

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model_output"
MAX_LENGTH = 384
STRIDE = 128
EPOCHS = 3
BATCH_SIZE = 16

# ─── Sample FAQ Data ────────────────────────────────────────────────────────────
# Replace with your own 10K+ Q&A pairs (SQuAD-format JSON).
SAMPLE_DATA = [
    {
        "context": "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging.",
        "question": "How many days do I have to return an item?",
        "answers": {"text": ["30 days"], "answer_start": [38]},
        "id": "q1",
    },
    {
        "context": "Technical support is available Monday through Friday, 9 AM to 6 PM EST. You can reach us via email or phone.",
        "question": "When is technical support available?",
        "answers": {"text": ["Monday through Friday, 9 AM to 6 PM EST"], "answer_start": [30]},
        "id": "q2",
    },
    {
        "context": "To reset your password, click on 'Forgot Password' on the login page and enter your registered email address.",
        "question": "How do I reset my password?",
        "answers": {"text": ["click on 'Forgot Password' on the login page"], "answer_start": [25]},
        "id": "q3",
    },
]


def load_dataset_from_file(path: str) -> DatasetDict:
    """Load SQuAD-format JSON. Falls back to sample data if not found."""
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        records = []
        for article in raw["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if not qa["is_impossible"]:
                        records.append({
                            "id": qa["id"],
                            "context": context,
                            "question": qa["question"],
                            "answers": {
                                "text": [a["text"] for a in qa["answers"]],
                                "answer_start": [a["answer_start"] for a in qa["answers"]],
                            },
                        })
        dataset = Dataset.from_list(records)
    else:
        print("⚠️  No dataset file found — using sample data.")
        dataset = Dataset.from_list(SAMPLE_DATA)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def preprocess(examples, tokenizer):
    """Tokenize and align answer spans."""
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]
        answer_start = answers["answer_start"][0]
        answer_end = answer_start + len(answers["text"][0])

        sequence_ids = inputs.sequence_ids(i)
        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # Answer not in this chunk → point to CLS
        if offsets[ctx_start][0] > answer_end or offsets[ctx_end][1] < answer_start:
            start_positions.append(0)
            end_positions.append(0)
        else:
            s = ctx_start
            while s <= ctx_end and offsets[s][0] <= answer_start:
                s += 1
            start_positions.append(s - 1)

            e = ctx_end
            while e >= ctx_start and offsets[e][1] >= answer_end:
                e -= 1
            end_positions.append(e + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def train(data_path: str = "data/train.json"):
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)

    raw_datasets = load_dataset_from_file(data_path)
    tokenized = raw_datasets.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DefaultDataCollator(),
        tokenizer=tokenizer,
    )

    print("🚀 Starting fine-tuning DistilBERT for Q&A...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
