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

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
def preprocess_tokens (tokens):
    list_of_wrong_letter = ['_', '«','»', '-']
    new_tokens_list = list()

    for sentence in tokens['tokens']:
        new_tokens = ' '.join(sentence)
        for letter in new_tokens:
            if letter in list_of_wrong_letter:
                for wrong_letter in list_of_wrong_letter:
                    new_tokens = new_tokens.replace(wrong_letter, ' ')
        list_words = new_tokens.split()  
        formated_list = [sentence for sentence in list_words] # Add 
        new_tokens_list.append(formated_list)
    return new_tokens_list

# %%

correct_strings = preprocess_tokens(dataset_train)

print (correct_strings)