# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import preprocess 

# %%
model_name = "neuralmind/bert-base-portuguese-cased"
model = AutoModel.from_pretrained(model_name)
url = 'liaad/Propbank-BR'
dataset = load_dataset(url, 'flatten')
fullDatasetTrain = dataset['train']
fullDatasetTest = dataset['test']
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
newTokensTrain = preprocess.preprocess_tokens(fullDatasetTrain)
newTokensTest = preprocess.preprocess_tokens(fullDatasetTest)

updatedDatasetTrain = fullDatasetTrain.remove_columns('tokens')
updatedDatasetTrain = updatedDatasetTrain.add_column('tokens', newTokensTrain)

updatedDatasetTest = fullDatasetTest.remove_columns('tokens')
updatedDatasetTest = updatedDatasetTest.add_column('tokens', newTokensTest)

# %%
def tokenize_and_align_labels (examples):
    tokenized_inputs = tokenizer (examples['tokens'], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f'frames']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx: 
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
    
# %%

tokenizedDataset = dataset.map(tokenize_and_align_labels, batched=True)
print (tokenizedDataset['train']['frames'])