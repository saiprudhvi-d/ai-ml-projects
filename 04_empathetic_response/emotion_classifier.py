"""
Project 04: Empathetic Response AI
Stage 1: Emotion classification via RoBERTa
Fine-tuned on GoEmotions dataset for 28 emotion categories.
"""

import os
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report
import numpy as np
import torch

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./roberta_emotion_output"
MAX_LENGTH = 128
EPOCHS = 4
BATCH_SIZE = 32

# GoEmotions 28 emotion labels → customer support subset
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
NUM_LABELS = len(EMOTION_LABELS)
ID2LABEL = {i: l for i, l in enumerate(EMOTION_LABELS)}
LABEL2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}


def load_data():
    try:
        dataset = load_dataset("go_emotions", "simplified")
        print(f"✅ Loaded GoEmotions: {dataset}")
        return dataset
    except Exception as e:
        print(f"⚠️  Could not load GoEmotions ({e}). Using synthetic data.")
        from datasets import Dataset, DatasetDict
        samples = [
            {"text": "I'm so frustrated, this is the third time this has happened!", "labels": [2]},   # anger
            {"text": "Thank you so much, you've been incredibly helpful!", "labels": [15]},             # gratitude
            {"text": "I don't understand what went wrong with my order.", "labels": [6]},              # confusion
            {"text": "This is unacceptable! I want a refund immediately.", "labels": [2]},             # anger
            {"text": "I'm a bit worried about the delivery timeline.", "labels": [14]},                # fear
            {"text": "Great service! Really happy with the resolution.", "labels": [17]},              # joy
        ] * 100
        ds = Dataset.from_list(samples)
        split = ds.train_test_split(test_size=0.15, seed=42)
        return DatasetDict({"train": split["train"], "test": split["test"]})


def preprocess(examples, tokenizer):
    tokens = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)
    # Binarize multi-label to single most prominent emotion
    labels = [l[0] if l else 27 for l in examples["labels"]]  # 27 = neutral
    tokens["labels"] = labels
    return tokens


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def train():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    raw = load_data()
    tokenized = raw.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=["text", "labels"])
    tokenized.set_format("torch")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("🚀 Fine-tuning RoBERTa for emotion classification...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Emotion classifier saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
