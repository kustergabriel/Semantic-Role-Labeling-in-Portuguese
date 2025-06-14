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
dataset_test = dataset['test']
srl_frames00 = dataset_train['srl_frames'][0][0]['frames']
print (srl_frames00)
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

# %%

# Tokenizer doing only for one sentence

def tokenize_and_align_labels (sentences,dataset_train):

    new_labels = list ()
    prev_word_id = None

    sentence00 = sentences[0]
    srl_frames00 = dataset_train['srl_frames'][0][0]['frames']

    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenized_sentence = tokenizer(sentence00, padding = True, is_split_into_words=True, return_offsets_mapping=True)

    words_ids = tokenized_sentence.word_ids()

    for word_id in words_ids:

        if word_id is None:
            new_labels.append('O')
        elif word_id != prev_word_id:
            new_labels.append(srl_frames00[word_id])
        else:
            new_labels.append('O')
        
        prev_word_id = word_id

    for words, labels in zip (words_ids,new_labels):
        print (words, labels)

tokenize_and_align_labels(new_tokens,dataset_train)

# %%

# Trying doing tokenizer for all frames

def tokenize_and_align_labels (sentences,dataset_train):

    tokenizer = AutoTokenizer.from_pretrained(model)
    new_labels = list ()
    prev_word_id = None
    frames_correct = list () # List of frames to align
    tokenized_sentence = list ()
    words_ids = list ()

    for frames in dataset_train['srl_frames']:
        for framess in frames:
            print (f"DEBUG 1: {framess}") # OK
            frames_correct.append (framess['frames'])


    #Tokenizing all sentences
    for sentences_to_tokenize in sentences:
        tokenized_sentence.append(tokenizer(sentences_to_tokenize, padding = True, is_split_into_words=True, return_offsets_mapping=True))

    for transform_to_word_ids in tokenized_sentence:
        words_ids.append(transform_to_word_ids.word_ids())

    for word_id in words_ids:
        print (word_id)

'''
    for word_id in words_ids:

        if word_id is None:
            new_labels.append('O')
        elif word_id != prev_word_id:
            new_labels.append(srl_frames00[word_id])
        else:
            new_labels.append('O')
        
        prev_word_id = word_id

    for words, labels in zip (words_ids,new_labels):
        print (words, labels)

'''

tokenize_and_align_labels(new_tokens,dataset_train)