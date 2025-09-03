# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import PreprocessData

# %%
model_name = "neuralmind/bert-base-portuguese-cased"
model = AutoModel.from_pretrained(model_name)
url = 'liaad/Propbank-BR'
datasetFlatten = load_dataset(url, 'flatten')
datasetDefault = load_dataset(url, 'default')
fullDatasetTrain = datasetFlatten['train']
fullDatasetTest = datasetFlatten['test']
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% 
# Mapear as roles -> Ex.: ARG0 -> 0, etc...

# Pegar os dois datasets.frames e ir juntando os valores

datasetFlattenFrames = datasetFlatten['train']['frames']
datasetDefaultFrames = datasetDefault['train']['srl_frames']

escritaArquivo = dict()
listaFramesDefault = list()
# Aqui percebi que default pode ter mais de um verbo na pos 1, e flatten é 1 verbo por posicao...
for framesGeraisDefault in datasetDefaultFrames:
    for framesEmCadaVerboDefault, framesEmCadaVerboFlatten in zip(framesGeraisDefault, datasetFlattenFrames):
        listaFramesDefault.append(framesEmCadaVerboDefault['frames'])

for framesD, framesF in zip(listaFramesDefault, datasetFlattenFrames):
    for d, f in zip(framesD, framesF):
        if d not in escritaArquivo:
            escritaArquivo[d] = framesF
            with open('arquivo.txt', 'a') as arquivo:
                arquivo.write(f'{d} - {f}\n')

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
newTokensTrain = PreprocessData.preprocess_tokens(fullDatasetTrain)
newTokensTest = PreprocessData.preprocess_tokens(fullDatasetTest)

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