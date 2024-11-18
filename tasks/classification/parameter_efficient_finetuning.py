import os
import torch
import numpy as np

from transformers import TrainingArguments, EvalPrediction
from transformers import BertTokenizer
from transformers import BertConfig
from datasets import load_dataset

from adapters import AutoAdapterModel
from adapters import AdapterTrainer

dataset = load_dataset("rotten_tomatoes")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

config = BertConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
)
model = AutoAdapterModel.from_pretrained(
    "bert-base-uncased",
    config=config,
)

# Add a new adapter
model.add_adapter("rotten_tomatoes", config="lora")
# Add a matching classification head
model.add_classification_head(
    "rotten_tomatoes",
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
)

model.train_adapter("rotten_tomatoes")
model.set_active_adapters("rotten_tomatoes")

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./experiments/bert_adapter_rotten_tomatoes",
    overwrite_output_dir=True,
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)
trainer.train()
trainer.evaluate()