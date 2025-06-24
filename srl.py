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

dataset_train = dataset['train']
dataset_test = dataset['test']
srl_frames00 = dataset_train['srl_frames'][0]
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

def tokenize_and_align_labels (sentences,dataset_train):

    tokenizer = AutoTokenizer.from_pretrained(model)
    new_labels_aligned = dict ()

    for index, sentences_to_tokenize in enumerate(sentences):
        prev_word_id = None
        tokenized_sentence = list ()
        new_labels = []

        tokenized_sentence = tokenizer(sentences_to_tokenize, 
                                            padding = True, 
                                            is_split_into_words=True, 
                                            return_offsets_mapping=True)
        
        words_ids = tokenized_sentence.word_ids()

        frames_for_this_sentence = dataset_train['srl_frames'][index]
        # print (frames_for_this_sentence)
        
        for word_id in words_ids:
            if word_id is None:
                new_labels.append('O')
            elif word_id != prev_word_id:
                if word_id < len(frames_for_this_sentence):
                    new_labels.append(frames_for_this_sentence[word_id])
                else:
                    new_labels.append('O')  # word_id inválido para os frames
            else:
                new_labels.append('O')
                
            prev_word_id = word_id

        new_labels_aligned[index] = {'frames': new_labels}

        print (new_labels_aligned)

tokenize_and_align_labels(new_tokens,dataset_train)