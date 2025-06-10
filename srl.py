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
print (dataset_train['srl_frames'][0])

# %%

# Tratando alguns caracteres que podem dar erro na tokenizacao
def preprocess_tokens (tokens):
    list_of_wrong_letter = ['_', '«','»']
    new_tokens_list = list()

    for sentence in tokens['tokens']:
        new_tokens = ' '.join(sentence)
        for letter in new_tokens:
            if letter in list_of_wrong_letter:
                for wrong_letter in list_of_wrong_letter:
                    new_tokens = new_tokens.replace(wrong_letter, '')
        list_words = new_tokens.split()  
        formated_list = [sentence for sentence in list_words]
        new_tokens_list.append(formated_list)
    return new_tokens_list

new_tokens = preprocess_tokens(dataset_train)
sentence00 = new_tokens[0]
print (sentence00)
# %%

# Tokenizer
def starting_to_tokenize ():
    
    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenized_sentence = tokenizer(sentence00, padding = True, is_split_into_words=True)

    word_ids = tokenized_sentence.word_ids() # Current labels

    tokenized_sentence_in_number = tokenized_sentence['input_ids']

    tokenized_sentence_in_word = tokenizer.convert_ids_to_tokens(tokenized_sentence_in_number)

    for word_ids, tokens in zip(word_ids,tokenized_sentence_in_word):
        print (word_ids,tokens)

# %%

# Align labels with tokenizers it's necessary because in tokenizer bert change unknown words for more words with ##    

# Run for all sentences in the list (new_tokens)

# Tokenize sentence and run labels for align with the sentence


# %%