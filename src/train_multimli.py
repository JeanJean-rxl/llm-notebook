from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Patch torch.load to always use weights_only=False
orig_torch_load = torch.load
def torch_load_weights_only_false(*args, **kwargs):
    kwargs["weights_only"] = False
    return orig_torch_load(*args, **kwargs)
torch.load = torch_load_weights_only_false


# 1. Load tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=3)

# 2. Load MultiNLI dataset
dataset = load_dataset("multi_nli")

# 3. Tokenize
def preprocess(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess, batched=True)

# 4. Set training arguments: use a batch size of 24, learning rate of 1e-5, and 5 training epochs for BERT
training_args = TrainingArguments(
    output_dir="./results_large_multinli",
    num_train_epochs=5,  
    per_device_train_batch_size=24,  
    learning_rate=1e-5,  
    evaluation_strategy="epoch",
    save_total_limit=3,         # Limit the total amount of checkpoints. Deletes the older checkpoints.
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation_matched"],
)

# 6. Train
# trainer.train()
trainer.train(resume_from_checkpoint="./results_large_multinli/checkpoint-55500")

# 7. Save
model.save_pretrained("./bert_large_multinli_finetuned")
# tokenizer.save_pretrained("./bert_large_multinli_finetuned")