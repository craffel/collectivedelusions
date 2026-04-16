import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import numpy as np

TASKS = ["mrpc", "cola", "rte", "qnli", "sst2"]
MODEL_ID = "t5-base"

def preprocess_function(examples, tokenizer, task):
    if task == "cola":
        inputs = ["cola sentence: " + doc for doc in examples["sentence"]]
    elif task == "sst2":
        inputs = ["sst2 sentence: " + doc for doc in examples["sentence"]]
    elif task in ["mrpc", "rte", "qnli"]:
        s1_key = "sentence1" if task != "qnli" else "question"
        s2_key = "sentence2" if task != "qnli" else "sentence"
        inputs = [f"{task} sentence1: " + s1 + " sentence2: " + s2 for s1, s2 in zip(examples[s1_key], examples[s2_key])]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    if task == "cola":
        label_map = {0: "unacceptable", 1: "acceptable"}
    elif task == "sst2":
        label_map = {0: "negative", 1: "positive"}
    elif task == "mrpc":
        label_map = {0: "no", 1: "yes"}
    elif task == "rte":
        label_map = {0: "entailment", 1: "not_entailment"}
    elif task == "qnli":
        label_map = {0: "entailment", 1: "not_entailment"}
        
    labels = [label_map[l] for l in examples["label"]]
    labels = tokenizer(text_target=labels, max_length=16, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_task(task):
    print(f"Training on {task}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    
    dataset = load_dataset("glue", task)
    # Only keep train and validation
    dataset = {split: dataset[split] for split in ["train", "validation"]}
    
    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, task), 
            batched=True,
            remove_columns=dataset[split].column_names
        )
    
    training_args = TrainingArguments(
        output_dir=f"./experts/{task}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        logging_steps=100,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained(f"./experts/{task}_final")
    tokenizer.save_pretrained(f"./experts/{task}_final")

if __name__ == "__main__":
    os.makedirs("./experts", exist_ok=True)
    for task in TASKS:
        if os.path.exists(f"./experts/{task}_final"):
            print(f"Expert for {task} already exists, skipping.")
            continue
        train_task(task)
