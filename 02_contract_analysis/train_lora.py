"""
Project 02: Contract Analysis AI
Fine-tunes LLaMA-7B with LoRA/PEFT on 5K+ legal documents
for clause extraction and summarization.
"""

import os
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import torch

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # Set HF_TOKEN env var for access
OUTPUT_DIR = "./lora_output"
MAX_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4

# ─── Sample contract data (replace with your 5K+ legal docs) ──────────────────
SAMPLE_CONTRACTS = [
    {
        "instruction": "Extract all termination clauses from this contract.",
        "input": "This Agreement shall terminate automatically upon the occurrence of any of the following: (a) mutual written agreement of both parties; (b) material breach by either party not cured within 30 days of written notice; (c) insolvency or bankruptcy of either party.",
        "output": "Termination clauses: (1) Mutual written agreement; (2) Material breach uncured after 30-day notice period; (3) Insolvency or bankruptcy of either party.",
    },
    {
        "instruction": "Summarize the payment terms in this contract.",
        "input": "The Client shall pay Consultant a monthly retainer of $10,000, due on the first business day of each month. Late payments shall accrue interest at 1.5% per month. The Client shall reimburse all pre-approved expenses within 15 days of invoice.",
        "output": "Payment: $10,000/month retainer due on the 1st business day. Late payments incur 1.5%/month interest. Approved expenses reimbursed within 15 days.",
    },
    {
        "instruction": "Identify the confidentiality obligations.",
        "input": "Each party agrees to maintain the confidentiality of the other party's Confidential Information for a period of five (5) years from the date of disclosure, using the same degree of care it uses to protect its own confidential information, but no less than reasonable care.",
        "output": "Confidentiality: 5-year obligation post-disclosure. Standard: reasonable care, same as protecting own confidential information.",
    },
]


def format_prompt(example: dict) -> str:
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def load_data(path: str = "data/contracts.json") -> DatasetDict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        print("⚠️  No dataset found — using sample data.")
        data = SAMPLE_CONTRACTS

    dataset = Dataset.from_list([{"text": format_prompt(d)} for d in data])
    split = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def train():
    # 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw = load_data()
    tokenized = raw.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("🚀 Fine-tuning LLaMA-7B with LoRA on contract data...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
