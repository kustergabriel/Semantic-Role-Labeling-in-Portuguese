# %%
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from huggingface_hub import login
import AlignLabels
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# %%
id2label = {
    0: "B-A0", 1: "B-A1", 2: "B-A2", 3: "B-A3", 4: "B-A4", 5: "B-AM-ADV",
    6: "B-AM-CAU", 7: "B-AM-DIR", 8: "B-AM-DIS", 9: "B-AM-EXT", 10: "B-AM-LOC",
    11: "B-AM-MNR", 12: "B-AM-NEG", 13: "B-AM-PNC", 14: "B-AM-PRD", 15: "B-AM-REC",
    16: "B-AM-TMP", 17: "B-C-A0", 18: "B-C-A1", 19: "B-C-A2", 20: "B-C-A3",
    21: "B-C-AM-ADV", 22: "B-C-AM-CAU", 23: "B-C-AM-DIS", 24: "B-C-AM-EXT",
    25: "B-C-AM-LOC", 26: "B-C-AM-MNR", 27: "B-C-AM-NEG", 28: "B-C-AM-PRD",
    29: "B-C-AM-TMP", 30: "B-C-V", 31: "B-V", 32: "I-A0", 33: "I-A1", 34: "I-A2",
    35: "I-A3", 36: "I-A4", 37: "I-AM-ADV", 38: "I-AM-CAU", 39: "I-AM-DIR",
    40: "I-AM-DIS", 41: "I-AM-EXT", 42: "I-AM-LOC", 43: "I-AM-MNR", 44: "I-AM-NEG",
    45: "I-AM-PNC", 46: "I-AM-PRD", 47: "I-AM-REC", 48: "I-AM-TMP", 49: "I-C-A0",
    50: "I-C-A1", 51: "I-C-A2", 52: "I-C-A3", 53: "I-C-AM-ADV", 54: "I-C-AM-CAU",
    55: "I-C-AM-LOC", 56: "I-C-AM-MNR", 57: "I-C-AM-PRD", 58: "I-C-AM-TMP",
    59: "I-C-V", 60: "O"
}
label2id = {v: k for k, v in id2label.items()}

modelFineTuning = AutoModelForTokenClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id )

data_collator = DataCollatorForTokenClassification(AlignLabels.tokenizer)

small_train_dataset = AlignLabels.tokenizedDataset["train"].shuffle(seed=42).select(range(1000))
train_dataset = AlignLabels.tokenizedDataset["train"]
eval_dataset = AlignLabels.tokenizedDataset["test"]

def compute_metrics(p):
    predictions, labels = p
    # transforma logits em classes preditas
    predictions = predictions.argmax(-1)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds = []
        seq_labels = []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id != -100:  # ignora subtokens
                seq_preds.append(AlignLabels.label_list[p_id])
                seq_labels.append(AlignLabels.label_list[l_id])
        true_predictions.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

# %%
# Nao configurei, apenas peguei do hugging face
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch"
)

trainer = Trainer(
    model=modelFineTuning,
    args=training_args,
    train_dataset=small_train_dataset,  
    eval_dataset=eval_dataset,
    tokenizer=AlignLabels.tokenizer,   
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
trainer.train()