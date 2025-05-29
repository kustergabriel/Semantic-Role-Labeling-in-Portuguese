# %%
from datasets import load_dataset
from transformers import AutoTokenizer

# %%
# Bertimbau da neuralmind
model = 'neuralmind/bert-base-portuguese-cased' 
# Dados anotados prontos -> BIO, CONLL
url = 'liaad/Propbank-BR'
dataset = load_dataset(url, "default")
# %%
dataset_train = dataset['train'] # FineTuning 
dataset_test = dataset['test'] # Testar para saber os scores
print (dataset_train['tokens'][0])
print (dataset_train['srl_frames'][0])
for tokens,frames in zip(dataset_train['tokens'][0],dataset_train['srl_frames'][0]):
    print(tokens,frames[0])

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
list_of_words = ['_', '«','»', '-']
new_tokens_list = list()

for sentence in dataset_train['tokens']:
    new_tokens = ' '.join(sentence)
    for word in new_tokens:
        if word in list_of_words:
            for wrong_word in list_of_words:
                new_tokens = new_tokens.replace(wrong_word, ' ')
    new_tokens_list.append(new_tokens)

print (new_tokens_list) 
# %%

