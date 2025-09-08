# %%
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from huggingface_hub import login
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import AlignLabels as al

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

data_collator = DataCollatorForTokenClassification(al.tokenizer)

small_train_dataset = al.tokenizedDatasetTrain
eval_dataset = al.tokenizedDatasetTest

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = []
    true_preds = []

    for l, p in zip(labels, preds):
        temp_labels = []
        temp_preds = []
        for li, pi in zip(l, p):
            if li != -100:  # ignora subwords/padding
                temp_labels.append(id2label[li])
                temp_preds.append(id2label[pi])
        true_labels.append(temp_labels)
        true_preds.append(temp_preds)

    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)

    print(classification_report(true_labels, true_preds, digits=4))

    return {"precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,            
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2, 
    num_train_epochs=8,            
    weight_decay=0.01,
    warmup_ratio=0.1,              
    lr_scheduler_type="linear",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"               
)

'''
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch"
) Resultado melhor ate agora -> 0.416426
'''
trainer = Trainer(
    model=modelFineTuning,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=al.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
trainer.train()