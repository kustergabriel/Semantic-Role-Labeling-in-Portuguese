# %%
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from huggingface_hub import login
import srl.AlignLabels as AlignLabels
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score



modelFineTuning = AutoModelForTokenClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=61)
data_collator = DataCollatorForTokenClassification(AlignLabels.tokenizer)

small_train_dataset = AlignLabels.tokenizedDataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = AlignLabels.tokenizedDataset["test"]

# Nao usei essa parte aqui
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_predictions = [
        [AlignLabels.label_list[pred] for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [AlignLabels.label_list[label] for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]

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
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=modelFineTuning,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    processing_class=AlignLabels.tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)

# %%
trainer.train()